'''
This Python3 script downloads the list of domains belonging to the specified Pfam family 
or CATH homologous superfamily and prints it in JSON in format { pdb: [[domain_name, chain, range]] }.

Example usage:
    python3  -m domains_from_pdbeapi  1.10.630.10
'''

import sys
import requests
import json
from typing import Optional

from .libs import lib
from .libs.lib_domains import Domain
from .libs.lib_cli import cli_command, run_cli_command

#  CONSTANTS  ################################################################################

DEFAULT_API_URL = 'https://www.ebi.ac.uk/pdbe/api/mappings'
DEFAULT_NUMBERING = 'label'

#  FUNCTIONS  ################################################################################

def create_multidict(key_value_pairs):  # Creates a dictionary with a list of multiple values per key. 
    multidict = {}
    for key, value in key_value_pairs:
        if not key in multidict:
            multidict[key] = []
        multidict[key].append(value)
    return multidict

def get_domain_name(mapping, default=None):
    return mapping.get('domain', default)

def get_start(mapping, numbering=DEFAULT_NUMBERING):
    return mapping['start']['author_residue_number'] if numbering == 'auth' else mapping['start']['residue_number']

def get_end(mapping, numbering=DEFAULT_NUMBERING):
    return mapping['end']['author_residue_number'] if numbering == 'auth' else mapping['end']['residue_number']

def get_chain(mapping, numbering=DEFAULT_NUMBERING) -> str:
    return mapping['chain_id'] if numbering == 'auth' else mapping['struct_asym_id']

def get_range(mapping, numbering=DEFAULT_NUMBERING) -> str:
    if numbering == 'auth':
        start = str(mapping['start']['author_residue_number'] or '') + mapping['start']['author_insertion_code']
        end = str(mapping['end']['author_residue_number'] or '') + mapping['end']['author_insertion_code']
    else:
        start = str(mapping['start']['residue_number'] or '')
        end = str(mapping['end']['residue_number'] or '')
    return start + ':' + end

def get_domains_multisegment(mappings, pdb, join_domains_in_chain=False, chain_change_warning=False):  # Returns a list of tuples (domain_name, chain, ranges).
    segments = []
    for mapping in mappings:
        domain_name = get_domain_name(mapping)
        chain = get_chain(mapping, 'label')
        rang = get_range(mapping, 'label')
        auth_chain = get_chain(mapping, 'auth')
        auth_rang = get_range(mapping, 'auth')
        if chain_change_warning and chain != auth_chain:
            sys.stderr.write(f'Warning: {pdb} {auth_chain} --> {chain}\n')
        segments.append(Domain(name=domain_name, pdb=pdb, chain=chain, ranges=rang, auth_chain=auth_chain, auth_ranges=auth_rang))
    if all(segment.name is not None for segment in segments):
        # All segments have a domain name, join segments with the same domain name.
        segments_by_domain = create_multidict((seg.name, seg) for seg in segments)
        result = []
        for domain_name, domain_segments in sorted(segments_by_domain.items()):
            chains = [seg.chain for seg in domain_segments]
            ranges = [seg.ranges for seg in domain_segments]
            auth_chains = [seg.auth_chain for seg in domain_segments]
            auth_ranges = [seg.auth_ranges for seg in domain_segments]
            chains = sorted(set(chains))
            auth_chains = sorted(set(auth_chains))
            if len(chains) > 1:
                sys.stderr.write('Error: Some domains contain parts of multiple chains\n')
                exit(1)
            chain = chains[0]
            auth_chain = auth_chains[0]
            ranges = ','.join(str(rang) for rang in ranges)
            auth_ranges = ','.join(str(rang) for rang in auth_ranges)
            result.append(Domain(name=domain_name, pdb=pdb, chain=chain, ranges=ranges, auth_chain=auth_chain, auth_ranges=auth_ranges))
    else:
        # Some segments miss a domain name, creating new domain names.
        ranges_by_chain = create_multidict( (seg.chain, seg) for seg in segments )
        result = []
        for chain, chain_segments in sorted(ranges_by_chain.items()):
            auth_chain = next(seg.auth_chain for seg in chain_segments)
            ranges = [seg.ranges for seg in chain_segments]
            auth_ranges = [seg.auth_ranges for seg in chain_segments]
            if join_domains_in_chain:
                ad_hoc_name = f'{pdb}_{chain}_0'
                ranges = ','.join( str(rang) for rang in ranges )
                auth_ranges = ','.join( str(rang) for rang in auth_ranges )
                result.append(Domain(name=ad_hoc_name, pdb=pdb, chain=chain, ranges=ranges, auth_chain=auth_chain, auth_ranges=auth_ranges))
            else:
                for i, (rang, auth_rang) in enumerate(zip(ranges, auth_ranges)):
                    ad_hoc_name = f'{pdb}_{chain}_{i}'
                    result.append(Domain(name=ad_hoc_name, pdb=pdb, chain=chain, ranges=rang, auth_chain=auth_chain, auth_ranges=auth_rang))
    return result

#  MAIN  ###############################################################################

@cli_command()
def main(accession: str, source: str = DEFAULT_API_URL, join_domains_in_chain: bool = False, chain_change_warning: bool = False) -> None:
    '''Download the list of domains belonging to the specified Pfam family or CATH homologous superfamily 
    and print it in JSON in format { pdb: [[domain_name, chain, range]] }.
    @param  `accession`              "accession" argument to PDBe API SIFTS Mappings, e.g. Pfam accession or CATH cathcode.
    @param  `source`                 URL with PDBeAPI server.
    @param  `join_domains_in_chain`  Join all domains in one chain if their names are not provided by the source.
    @param  `chain_change_warning`   Print warning if struct_asym_id != chain_id (label_ vs auth_ chain numbering.
    '''    
    output = {}

    if accession is None or accession == '':
        sys.stderr.write(f'Invalid accession: {repr(accession)}, producing empty domain list.\n')

    else:
        url = f'{source}/{accession}'
        sys.stderr.write('Downloading ' + url + '\n')
        response = requests.get(url)
        if response.ok:
            results = json.loads(response.text).get(accession, {}).get('PDB', {})
        elif response.status_code == 404:
            sys.stderr.write(f'HTTP error {response.status_code} ({url}). Assuming the family {accession} is empty.\n')
            results = {}
        else: 
            raise Exception(f'HTTP request failed, status code {response.status_code} ({url}).\n')

        for pdb, entry in sorted(results.items()):
            if isinstance(entry, list):
                mappings = entry
            elif isinstance(entry, dict) and 'mappings' in entry:
                mappings = entry['mappings']
            else:
                raise Exception(f'Did not find mappings in {url}')
            output[pdb] = get_domains_multisegment(mappings, pdb, join_domains_in_chain=join_domains_in_chain, chain_change_warning=chain_change_warning)

    n_pdbs = len(output)
    n_domains = sum( len(doms) for pdb, doms in output.items() )
    sys.stderr.write(f'Found {n_domains} domains in {n_pdbs} PDB entries.\n')
    lib.dump_json(output, sys.stdout)


if __name__ == '__main__':
    run_cli_command(main)
    