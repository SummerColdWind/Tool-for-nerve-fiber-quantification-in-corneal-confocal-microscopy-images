from utils.calculate import get_CNFL, get_CNFD, get_CNBD


def perform_analysis(window):
    if window.raw_image is not None:
        try:
            window.p.load_image(window.raw_image)
            window.p.process()
            print(f'[CNFL: {get_CNFL(window.p):.3f}]\t'
                  f'[CNFD: {get_CNFD(window.p):.3f}]\t'
                  f'[CNBD: {get_CNBD(window.p):.3f}]')
        except Exception as e:
            print(f"Error in analysis: {e}")
    else:
        print('Please load an image first.')
