from pathlib import Path
import shutil
import sys
import argparse


def make_archive(src: Path, dest: Path) -> Path:
    fmt = dest.suffix.lstrip('.')
    archive = shutil.make_archive(str(dest.with_suffix('')), fmt, str(src))
    return Path(archive)

def main():
    parser = argparse.ArgumentParser(description='Zip all input_dir/families/*/results --> output_dir/results-*.zip')
    parser.add_argument('input_dir', type=str)
    parser.add_argument('output_dir', type=str)
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fam_dirs = sorted(fam for fam in Path(input_dir, 'families').iterdir() if fam.is_dir)
    # print(fam_dirs)
    for fd in fam_dirs:
        family = fd.name
        try:
            make_archive(Path(fd, 'results'), Path(output_dir, f'results-{family}.zip'))
        except FileNotFoundError:
            print('Missing results/ directory:', family, file=sys.stderr)
    


if __name__ == "__main__":
    main()