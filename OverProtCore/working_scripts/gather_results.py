from pathlib import Path
import shutil
import sys
import argparse


def make_archive(src: Path, dest: Path) -> Path:
    fmt = dest.suffix.lstrip('.')
    archive = shutil.make_archive(str(dest.with_suffix('')), fmt, str(src))
    return Path(archive)

def main():
    parser = argparse.ArgumentParser(description='Zip all input_dir/families/*/results --> output_dir/results-*.zip; copy input_dir/families/*/results/consensus.png --> png_dir/*.png')
    parser.add_argument('input_dir', type=Path)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('-p', '--png_dir', type=Path, default=None)
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    png_dir = args.png_dir

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if png_dir is not None:
        Path(png_dir).mkdir(parents=True, exist_ok=True)

    fam_dirs = sorted(fam for fam in Path(input_dir, 'families').iterdir() if fam.is_dir)
    # print(fam_dirs)
    for fd in fam_dirs:
        family = fd.name
        try:
            make_archive(Path(fd, 'results'), Path(output_dir, f'results-{family}.zip'))
            shutil.copy(Path(fd, 'results', 'consensus.png'), Path(png_dir, f'{family}.png'))
        except FileNotFoundError:
            print('Missing results/ directory:', family, file=sys.stderr)
    


if __name__ == "__main__":
    main()
