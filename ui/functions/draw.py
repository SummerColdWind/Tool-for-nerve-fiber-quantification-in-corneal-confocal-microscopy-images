from process.draw import draw_result_image

def draw_bone(window):
    try:
        p = window.p
        result_image = draw_result_image(p.segments, p.nodes, p.image, 'none', 'bone')
        window.cur_image = result_image
        window.show_image()
    except Exception as e:
        print(f"Error when drawing: {e}")


def draw_body(window):
    try:
        p = window.p
        result_image = draw_result_image(p.segments, p.nodes, p.image, 'none', 'body')
        window.cur_image = result_image
        window.show_image()
    except Exception as e:
        print(f"Error when drawing: {e}")


def draw_binary(window):
    try:
        p = window.p
        result_image = p.images['binary']
        window.cur_image = result_image
        window.show_image()
    except Exception as e:
        print(f"Error when drawing: {e}")


def show_origin(window):
    try:
        p = window.p
        result_image = p.image
        window.cur_image = result_image
        window.show_image()
    except Exception as e:
        print(f"Error when drawing: {e}")
