// Converts color name ('grey', 'gray' will work) to HEX
function colorNameToHex(colorName){
	const ctx = document.createElement('canvas').getContext('2d');
	ctx.fillStyle = colorName;
	return ctx.fillStyle;
}

function hexToRgb(hex) {
	// in case it is not hex, e.g. gray/grey sometimes occur in 2DProts layout JSON
	if (hex[0] !== '#') {
		hex = colorNameToHex(hex);
	}
	let result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
	return result ? {
		r: parseInt(result[1], 16),
		g: parseInt(result[2], 16),
		b: parseInt(result[3], 16)
	} : null;
}

class IntegratedViewer {
	constructor() {
		this.plugins = {
			'twoD': {
				'instance': undefined,
			},
			'threeD': {
				'instance': undefined,
			},
		};
		this.wellcomePage = true;
		this.currentUrl = new URL(window.location.href);
		this.familyId;
		// the one that is used by overprot! not by 2dprots
		this.domainId;
		
		this.defaultFamilyId = '2.160.10.30';
		this.defaultDomainId = '3oh1A02';
		
		// this.twoDprotsDomainId;
		this.twoDProtsTimestamp;
		
		this.pdbId;
		this.structAsymId;
		this.authAsymId;
		this.entityId;
		
		this.familyAndDomainForm = document.getElementById('familyAndDomainForm');
		this.familiesSelectEle = document.getElementById('familiesSelector');
		this.domainsSelectEle = document.getElementById('domainsSelector');
		this.submitButton = document.getElementById('submitFamilyAndDomainForm');
		
		this.currentFamilyIdEle = document.getElementById('currentFamilyIdEle');
		this.currentDomainIdEle = document.getElementById('currentDomainIdEle');				
	}
	
	parseUrl() {
		this.familyId = this.currentUrl.searchParams.get('family_id');
		this.domainId = this.currentUrl.searchParams.get('domain_id');
		if (this.familyId && this.domainId) {
			this.wellcomePage = false;
		}
	}
	
	showCurrentFamilyAndDomain() {
		this.currentFamilyIdEle.innerHTML = `Current family (CATH): <strong>${this.familyId}</strong>`;
		this.currentDomainIdEle.innerHTML = `Current domain: <strong>${this.domainId}</strong>`;
	}
	
	async setData() {	
		// domains we need to get current domain data, ranges - to get entityId from chainId (=== chain_id(PDBeAPI))
		const domainsResponse = await fetch(`https://overprot.ncbr.muni.cz/data/db/families/${this.familyId}/domains.json`);
		const domains = await domainsResponse.json();
		
		const currentDomainData = domains.filter(d => d.domain === this.domainId)[0];
		this.pdbId = currentDomainData['pdb'];
		// in overprot API chain_id corresponds to struct_asym_id (PDBeAPI)
		this.structAsymId = currentDomainData['chain_id'];
		this.authAsymId = currentDomainData['auth_chain_id'];
		// assembling twoDprotsDomainId
		// Not necessary as 2DProts switched from struct_asym_id to chain_id
		// this.twoDprotsDomainId = `${this.domainId.slice(0, 4)}${this.structAsymId}${this.domainId.slice(5)}`;
		
		// Approach with parsing HTML from 2DProts still does not work for all, e.g. family_id=2.60.40.10&domain_id=12e8H01 gets correct timestamp, using which svg is accesible, but json is not (just changing .svg to .json at the end of link)
		// const [twoDprotsResponse, rangesResponse] = await Promise.all([
			// fetch(`https://2dprots.ncbr.muni.cz/domain/${this.twoDprotsDomainId}`),
			// fetch(`https://www.ebi.ac.uk/pdbe/api/pdb/entry/polymer_coverage/${this.pdbId}/chain/${this.authAsymId}`)
		// ]);
		
		// const twoDProtsHTML = await twoDprotsResponse.text();
		// const parser = new DOMParser();
		// const doc = parser.parseFromString(twoDProtsHTML, 'text/html');
		// console.log(doc);
		// const anchorWithSVGLink = doc.querySelector('a[href$=".svg"]');
		// this.twoDProtsTimestamp = anchorWithSVGLink.href.split('/').splice(-2, 1);
		
		const rangesResponse = await fetch(`https://www.ebi.ac.uk/pdbe/api/pdb/entry/polymer_coverage/${this.pdbId}/chain/${this.authAsymId}`);
		
		const ranges = await rangesResponse.json();
		if (ranges[this.pdbId].molecules.length > 1) {
			console.error('entityId determination based on chainId and entryId may be wrong: there is several molecules in ranges API response');
			console.log(ranges);
		}
		this.entityId = ranges[this.pdbId].molecules[0].entity_id;
	}
	
	populateSelectEle(ele, optionsArr) {
		ele[0].options.length = 0;
		var start = new Date();
		// With 30k+ options for some families it can freeze the selector in disabled state
		
		// 14k ms
		// for (const item of optionsArr) {
			// $(ele).append(new Option(item, item))
		// }
		
		// Possibly faster version
		// 10k ms
		// $(ele).append($.map(optionsArr, (item) => new Option(item, item)));
		
		// let html = '';
		// for (const item of optionsArr) {
			// html = html + `<option value="${item}">${item}</option>`;
		// }
		// ele.append(html);
		
		let html = '';
		// superoptimized
		for (var i = 0, len = optionsArr.length; i < len; ++i) {
			html = html + `<option value="${optionsArr[i]}">${optionsArr[i]}</option>`;
		}
		ele[0].innerHTML = html;
		
		var time = new Date() - start;
		// console.log(time);
		
		ele.selectpicker('refresh');
	}
	
	bindFormSubmitListener() {
		const _this = this;
		this.familyAndDomainForm.addEventListener('submit', (event) => {
			event.preventDefault();
			try {
				const requestedUrl = new URL(_this.currentUrl);
				requestedUrl.searchParams.set('family_id', _this.familiesSelectEle.value);
				requestedUrl.searchParams.set('domain_id', _this.domainsSelectEle.value);
				window.location.href = requestedUrl.href;
			} catch (error) {
				throw new Error(error.message);
			}
		});
	}
	
	render3D() {
		//Create plugin instance
		const viewerInstance = new PDBeMolstarPlugin();
		this.plugins.threeD.instance = viewerInstance;
		// console.log(viewerInstance);
		
		var options = {
			// moleculeId: this.pdbId,
			hideControls: true,
			customData: {
				url: `https://www.ebi.ac.uk/pdbe/coordinates/${this.pdbId}/chains?entityId=${this.entityId}&authAsymId=${this.authAsymId}&encoding=bcif`,
				format: 'cif',
				binary: true
			},
			pdbeLink: true,
			bgColor: {r:255, g:255, b:255},
		}
		// Alternative coordinate server: https://cs.litemol.org/ (same syntax)

		//Get element from HTML/Template to place the viewer 
		var viewerContainer = document.getElementById('myViewer');

		//Call render method to display the 3D view
		viewerInstance.render(viewerContainer, options);
		
		// Promise for 3D load completion
		const loadComplete = new Promise((resolve, reject) => {
			viewerInstance.events.loadComplete.subscribe(() => {resolve()});
			console.log('3d loaded');
		});
		
		const twoDProtsApiDataObtained = new Promise((resolve, reject) => {
			document.addEventListener('PDBtopologyViewerApiDataLoaded', resolve, false);
		});
		
		const _this = this;
		Promise.all([loadComplete, twoDProtsApiDataObtained]).then(() => {
			// here we could hide loader for 2dprots or for all viewers
			console.log('both api data and 3d are ready')
			// Now all SSEs include coils data as well - just to color them on 3D according to coils colors on 2D view
			const allSSEs = _this.plugins.twoD.instance.twoDProtsData.topologyData.map((sse) => {
				// coloring according to 2DProts colors
				// console.log(hexToRgb(sse.color))
				return {
					auth_asym_id: _this.authAsymId,
					color: hexToRgb(sse.color),
					// focus: true,
					start_residue_number: sse.start,
					end_residue_number: sse.stop,
				}
			});
			// console.log(allSSEs);
			const targetChainSelection = [{
				// technically struct_asym_id corresponds to _this.structAsymId, and auth_asym_id - to _this.authAsymId, but better to uniformly use one everywhere, and TopologyViewer event returns only auth...
				// struct_asym_id: _this.structAsymId,
				// as auth_asym_id (mmCIF) === chain_id (PDBeAPI) === chainId internally used in TopologyViewer code
				auth_asym_id: _this.authAsymId,
				// color:{r:255,g:255,b:255},
				// focus: true,
				// For selecting just the domain of interest
				start_residue_number: _this.plugins.twoD.instance.twoDProtsData.residueNumbers.start,
				end_residue_number: _this.plugins.twoD.instance.twoDProtsData.residueNumbers.stop,
			}];
			// Works, but rather slow, and anyway cuts repr parts due to 'focus effect'
			// viewerInstance.visual.select({data: targetChainSelection, nonSelectedColor: {r:16,g:16,b:16}}).then(() => {
				// viewerInstance.visual.select({ data: allSSEs, nonSelectedColor: {r:16,g:16,b:16}})
			// });
			
			viewerInstance.visual.select({ data: allSSEs, nonSelectedColor: {r:64,g:64,b:64}}).then(() => {
				viewerInstance.visual.focus(allSSEs);
			});
			
		})
		
		// Upon user hover on Topology Component residue, highlight that residue in MolStar view
		document.addEventListener('PDB.topologyViewer.mouseover', (event) => {
			viewerInstance.visual.highlight({ data: [{
				// entity_id: event.eventData['entityId'],
				// as auth_asym_id (mmCIF) === chain_id (PDBeAPI) === chainId internally used in TopologyViewer code
				auth_asym_id: event.eventData['chainId'],
				residue_number: event.eventData['residueNumber'],
			}], color: {r:255,g:255,b:255},});
		});
		
		document.addEventListener('PDB.topologyViewer.mouseout', (event) => {
			viewerInstance.visual.clearHighlight();
		});
		
		document.addEventListener('PDB.topologyComponent.click', (event) => {
		});
		
		// H5 helix is not present in 2D, so it is not highlighted
		document.addEventListener('PDB.overprot.hover', (event) => {
			if (event.detail.sses[0]) {
				// console.log('PDB.overprot.hover', event);
				const sses = event.detail.sses;
				
				let molstarData = sses.map(sse => {
					// Right now sseLabel selects 4 topo elements (two in mask, one outside of mask (default one), one which is raised above connecting coils). Fixed by adding class to the last one
					const data = d3.select(`#${sse.label}.topologyEleTopLayer`)
					.attr('fill', 'rgb(211,211,211)')
					.attr('fill-opacity','1.0')
					.data()
					// data[0] contains data bound to topologyEle, including residues
					// console.log(data[0]);
					
					// Checking if this SSE exists in 2D view
					if (data[0]) {
						const molstarSelectionData = {
							entity_id: data[0].proteinData.entityId.toString(),
							// as auth_asym_id (mmCIF) === chain_id (PDBeAPI) === chainId internally used in TopologyViewer code
							auth_asym_id: data[0].proteinData.chainId,
							start_residue_number: data[0].start,
							end_residue_number: data[0].stop,
						};
						return molstarSelectionData;
					} else {
						console.log(`${sse.label} secondary structure element is not present in selected domain, only in some other proteins of that family.`)
						return {}
					}					
				});
				
				molstarData = molstarData.filter(d => Object.keys(d).length !== 0);
				// console.log(molstarData);
				
				viewerInstance.visual.highlight({ data: molstarData, color: {r:255,g:255,b:255},});
				
			} else {
				// empty detail.label arr => event was fired onmouseover/offhover
				d3.selectAll('.topologyEle.topologyEleTopLayer')
				.attr('fill', 'none')
				.attr('fill-opacity', null)
				
				viewerInstance.visual.clearHighlight();
			}
		});
	}
		
	render2D() {
		// document.addEventListener('DOMContentLoaded', () => {
			//Create plugin instance
			const pluginInstance = new PdbTopologyViewerPlugin();
			this.plugins.twoD.instance = pluginInstance;
			//Get element from HTML/Template to place the view
			const viewContainer = document.getElementById('pdb-topology-viewer');
			// console.log(this.pdbId, this.entityId, this.authAsymId, this.familyId, this.domainId)
			const options = {
			  entryId: this.pdbId,
			  entityId: this.entityId,
			  // as auth_asym_id (mmCIF) === chain_id (PDBeAPI) === chainId internally used in TopologyViewer code
			  chainId: this.authAsymId,
			  // structAsymId should be provided as well as we actually need to request 2DProts layout data based on structAsymId from withing TopologyViewer, BUT TopologyViewer itself uses auth_asym_id === chain_id === chainId (internally)
			  structAsymId: this.structAsymId,
			  familyId:	this.familyId,
			  domainId: this.domainId,
			  twoDProtsTimestamp: this.twoDProtsTimestamp,
			  // entryId: '1akd',
			  // entityId: '1',
			  // chainId: 'A',
			  // entryId: '3oh1',
			  // entityId: '1',
			  // chainId: 'A',
			} 
		
			//Call render method to display the 2D view
			pluginInstance.render(viewContainer, options);
			
			// 2D => 1D hover event listener
			document.addEventListener('PDB.topologyViewer.mouseover', (event) => {
				document.querySelector('overprot-viewer').dispatchEvent(new CustomEvent('PDB.overprot.do.hover', {
					detail: {
								'sses': [{'label': event.eventData.parentSSEId}]
							}
						}));
			});
			
			document.addEventListener('PDB.topologyViewer.mouseout', (event) => {
				document.querySelector('overprot-viewer').dispatchEvent(new CustomEvent('PDB.overprot.do.hover', {
					detail: {
								'sses': []
							}
						}));
			});
	}
	
	render1D() {
		// Actually insert in HTML rather than render
		const html = `<overprot-viewer id='anything' file='https://overprot.ncbr.muni.cz/data/db/diagrams/diagram-${this.familyId}.json' width=1800 height=200 color-method='rainbow' shape-method='symcdf' beta-connectivity='on' occurrence-threshold='25%' dispatch-events='true' listen-events='true'></overprot-viewer>`;
		
		const container = document.getElementById('overprot-wrapper');
		container.innerHTML = html;
	}
	
	async loadFamiliesSelectOptions() {
		console.log('families request sent');
		// Get txt from overprot API
		const response = await fetch('https://overprot.ncbr.muni.cz/data/db/families.txt');
		const txt = await response.text();
		console.log('families request parsed');
		// Parse txt to get arr with family IDs
		const options = txt.replace(/\r/g, "").split(/\n/);
		// Populate familiesSelectEle with that arr
		const ele = $(this.familiesSelectEle);
		this.populateSelectEle(ele, options);
		console.log('families select populated');
		return ele;
	}
	
	async loadDomainsSelectOptions(familyId) {
		// Get json from overprot API
		console.log('domains request sent');
		const responseOverprotDomains = await fetch(`https://overprot.ncbr.muni.cz/data/db/families/${familyId}/domains.json`);
		const overprotDomains = await responseOverprotDomains.json();
		console.log('domains request parsed');
		// Potentially both domain lists from Overprot and 2DProts are consistent
		// If not, implement these: get two lists, and remove from Overprot list whatever is not in 2DProts list
		// You need to also request for a timestamp for that family before requesting the list from 2DProts
		// Plan:
		// 1. parallel request via Promise.all for overprot list and 2DProts HTML for that family
		// 2. parsing 2DProts HTML to get timestamp for that family
		// 3. request to 2DProts domain_list for that family using obtained timestamp
		// 4. remove from Overprot list whatever is not in 2DProts list
		// TRY CATCH - if cannot find domain_list, leave options as from overprot
		// const responseTwoDProtsDomains = await fetch(`https://2dprots.ncbr.muni.cz/static/web/generated-${familyId}/2021-10-04T11_52_33_653629990_02_00/domain_list.txt`);
		// const twoDProtsDomains = await responseTwoDProtsDomains.text();
		// const twoDProtsOptions = twoDProtsDomains.replace(/\r/g, "").split(/\n/);
		
		// Parse json to get arr with domain IDs
		const overprotOptions = overprotDomains.map(domain => domain.domain);
		
		// Populate domainsSelectEle with that arr
		const ele = $(this.domainsSelectEle);
		this.populateSelectEle(ele, overprotOptions);
		console.log('domains select populated');
		return ele;
	}
}

$.LoadingOverlaySetup({
	background      : "rgba(0, 0, 0, 0.8)",
	image           : '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000"><circle r="80" cx="500" cy="90"/><circle r="80" cx="500" cy="910"/><circle r="80" cx="90" cy="500"/><circle r="80" cx="910" cy="500"/><circle r="80" cx="212" cy="212"/><circle r="80" cx="788" cy="212"/><circle r="80" cx="212" cy="788"/><circle r="80" cx="788" cy="788"/></svg>',
	imageAnimation  : "2.0s rotate_right",
	imageColor      : "#FAFAFA",
	text			: "Loading domain data...",
	textColor 		: "#D3D3D3",
	textResizeFactor: 0.3,
});

const instance = new IntegratedViewer();
instance.parseUrl();
if (!instance.wellcomePage) instance.showCurrentFamilyAndDomain();
instance.bindFormSubmitListener();

// on programmatic or user change of families selector, load domains select options and set value (either [0] or instance.domainId if page is with query params)
$(instance.familiesSelectEle).on('changed.bs.select', function (e, clickedIndex, isSelected, previousValue) {
	// console.log(e, clickedIndex, isSelected, previousValue);
	instance.submitButton.disabled = true;
	const ele = $(instance.domainsSelectEle);
	$.LoadingOverlay('show');
	ele.attr('disabled', true).selectpicker('refresh');
	const selectedFamilyId = $(this).selectpicker('val');
	instance.loadDomainsSelectOptions(selectedFamilyId).then(() => {
		// if it was programmatic change && if it is a wellcome page OR if it was the user's change => select first domain in the list
		if ((clickedIndex === null && instance.wellcomePage) || clickedIndex !== null) {
			// select [0] and refresh.....
			try {
				// Here it breaks if the user clicks on family selector many times and too fast: "... prop 'value' of undefined"
				// E.g. 6.10.160.10 has empty domain list: https://overprot.ncbr.muni.cz/data/db/families/6.10.160.10/domains.json
				// Or maybe we should just disable family selector as well until everything is ready? Seems not.						
				console.log('selecting default domain option');
				const defaultOption = ele[0].options[0].value;
				ele.selectpicker('val', defaultOption);
				instance.submitButton.disabled = false;
				ele.attr('disabled', false);
				ele.selectpicker('refresh');
				ele.selectpicker('refresh');
			} catch(error) {
				console.log(error);
				alert('An error occured. Most likely the selected family is empty (i.e. has no domains). Please select another family.');
			}
		// if it was a programmatic change (on page with query params we set selectors values equal to that params)
		} else if (clickedIndex === null && !instance.wellcomePage) {
			// select instance.domainId and refresh.....
			console.log('selecting current domain option');
			ele.selectpicker('val', instance.domainId);
			instance.submitButton.disabled = false;
			ele.attr('disabled', false);
			ele.selectpicker('refresh');
			ele.selectpicker('refresh');
			// $.LoadingOverlay('hide');
			// console.log('select instance.domainId', ele, instance.domainId);
		} else {
			console.log('Something wrong happened');
			console.log(e, clickedIndex, isSelected, previousValue);
		}
		setTimeout(() => {
			$.LoadingOverlay('hide');
		}, 1000)
	});
});

instance.loadFamiliesSelectOptions().then(() => {
	const ele = $(instance.familiesSelectEle);
	if (instance.wellcomePage) {
		// select [0] and refresh.....
		// const defaultOption = ele[0].options[0].value;
		const defaultOption = instance.defaultFamilyId;
		ele.selectpicker('val', defaultOption);
		ele.selectpicker('refresh');
		ele.selectpicker('refresh');
		// console.log('select [0]', ele, defaultOption);
	} else {
		// select instance.familyId and refresh.....
		ele.selectpicker('val', instance.familyId);
		ele.selectpicker('refresh');
		ele.selectpicker('refresh');
		// console.log('select instance.domainId', ele, instance.domainId);
	}
})

if (!instance.wellcomePage) {
	document.addEventListener('DOMContentLoaded', () => {
		console.log('dom loaded')
		instance.setData()
		.then(() => {
			instance.render1D();
			instance.render2D();
			instance.render3D();
		});
	});
}		