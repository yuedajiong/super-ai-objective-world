import torch

class World:  #red:y=x^2, green:y=sin(x), blue:x＝a+r*cosθ,y＝b+r*sinθ
    def __init__(self):
        def create(points=50*2+1, size_h=256, size_w=256, do_save=0, do_show=0):
            def make_function(function, points):
                if function=='power':
                    x = torch.linspace(-1, +1, points)
                    y = (x ** 2)
                elif function=='trigonometric':
                    x = torch.linspace(-2*torch.pi, +2*torch.pi, points)
                    y = torch.sin(x)
                elif function=='circle':
                    theta = torch.linspace(0, 2*torch.pi, points)
                    radius = 1.
                    x = 0. + radius * torch.cos(theta)
                    y = 0. + radius * torch.sin(theta)
                else:
                    raise  
                return x, y

            def move_align(x, y, size_h, size_w, resize_factor=0.666):
                x_factor = (max(x)-min(x)) / 2.
                y_factor = (max(y)-min(y)) / 2.
                factor = x_factor if x_factor>y_factor else y_factor
                x = x * resize_factor / factor
                y = y * resize_factor / factor
                x = ((size_w / 2.) + (x * (size_w / 2.))).int().tolist()
                y = ((size_h / 2.) + (y * (size_h / 2.))).int().tolist()
                return x, y

            import torchvision
            def load_background(file, size_h, size_w):
                if file:
                    background = torchvision.io.read_image(file, mode=torchvision.io.image.ImageReadMode.RGB)
                    background = background.float()/255.0 
                    background = torch.nn.functional.interpolate(background.unsqueeze(0), size=(size_h, size_w), mode='bilinear', align_corners=False).squeeze(0)
                else:
                    background = torch.ones((3, size_h, size_w))*0.333
                return background

            def make_snapshot(xy, background, step, colors=torch.tensor([[1,0,0],[0,1,0],[1,0,1]]), keep_tail=True):
                def set_pixle(keep_tail, step, snapshot, size_h, y, x, color):
                    if keep_tail:
                        for s in range(step):
                            snapshot[:, size_h-y[s],x[s]] = color
                    snapshot[:, size_h-y[step]+0,x[step]+0] = color
                    snapshot[:, size_h-y[step]-1,x[step]-1] = color
                    snapshot[:, size_h-y[step]-1,x[step]+1] = color
                    snapshot[:, size_h-y[step]+1,x[step]-1] = color
                    snapshot[:, size_h-y[step]+1,x[step]+1] = color

                snapshot = background.clone()        
                for i in range(len(xy)):
                    x,y = xy[i]
                    color = colors[i%(len(colors))]                
                    set_pixle(keep_tail, step, snapshot, size_h, y, x, color)
                return snapshot

            def live_snapshot(xy, background, points):
                images = []
                for step in range(points):
                    snapshot = make_snapshot(xy, background, step)
                    images.append(snapshot.permute(1,2,0))  #HWC
                return images          

            def save_animation(images, filename='world_animation.gif'):
                import imageio; imageio.mimsave(filename, [(image*255).byte() for image in images], format='GIF-PIL', quantizer=1)

            def show_animation(images, pause_time=0.001):
                import matplotlib.pyplot as plt
                plt.axis('off')
                for step,image in enumerate(images):
                    plt.clf()
                    plt.imshow(image)
                    plt.pause(pause_time)  
                plt.pause(9)

            background = load_background(file=[None,'superi-cv-vision-to-symbol-world-background.png'][1], size_h=size_h, size_w=size_w)
            xy = []
            for function in ['power','trigonometric','circle'][:]:
                x, y = make_function(function=function, points=points)
                x, y = move_align(x=x, y=y, size_h=size_h, size_w=size_w)
                xy.append((x,y))
            images = live_snapshot(xy=xy, background=background, points=points)
            if do_save: save_animation(images)
            if do_show: show_animation(images)
            return images

        self.images = create()

    def obverse(self, batch_size):
        import random  
        start = random.randint(0, len(self.images)-batch_size-1)
        I, T = [], []
        for i in range(batch_size):
            I.append(self.images[start+i])
            T.append(self.images[start+i+1])
        return torch.stack(I, dim=0), torch.stack(T, dim=0)
