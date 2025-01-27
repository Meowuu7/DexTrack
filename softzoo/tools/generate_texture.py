import argparse
from PIL import ImageEnhance
from texturize import api, commands, io
from texturize.logger import ConsoleLog


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--resize-input', type=int, nargs='+', default=None)
    parser.add_argument('--output-size', type=int, nargs='+', default=[512, 512])
    parser.add_argument('--model', type=str, nargs='+', default='VGG11',
                        choices=['VGG19', 'VGG16', 'VGG13', 'VGG11', 'ThinetSmall', 'ThinetTiny'])
    parser.add_argument('--quality', type=float, default=2)
    parser.add_argument('--brightness', type=float, default=1.)
    parser.add_argument('--quiet', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()

    assert len(args.output_size) == 2, 'output size must be a 2-dim vector'

    # Load image
    image = io.load_image_from_file(args.input_path)
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(args.brightness)
    if args.resize_input is not None:
        assert len(args.resize_input) == 2, 'resize shape must be a 2-dim vector'
        image = image.resize(args.resize_input)

    # Coarse-to-fine synthesis runs one octave at a time.
    remix = commands.Remix(image)
    for result in api.process_octaves(remix, 
                                      octaves=5, 
                                      size=args.output_size,
                                      model=args.model,
                                      quality=args.quality,
                                      log=ConsoleLog(quiet=args.quiet, verbose=args.verbose)):
        pass

    # Save image
    io.save_tensor_to_file(result.images, args.output_path)


if __name__ == '__main__':
    main()
