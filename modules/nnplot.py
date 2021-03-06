import matplotlib.pyplot as plt
import matplotlib.animation as anm
from mpl_toolkits.mplot3d import axes3d
import tensorflow as tf
import numpy as np
import cv2
import os, shutil
import utility as ut

def nn_sol(nn_sol, space, t, x_lim, y_lim, file_path, fig_size=(8, 8), fps=24, num_pts=100, interval = 30):
    t = np.linspace(t[0], t[1], num=(t[1] - t[0]) * fps)
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    line, = ax.plot([], [])
    x = tf.convert_to_tensor(np.linspace(space[0][0], space[0][1], num=50), dtype=tf.float32)
    line.set_xdata(x)
    x = tf.reshape(x, shape=(num_pts, 1))
    def animate(frame):
        tt = t[frame] * tf.ones_like(x)
        y = nn_sol(tt, x).numpy()
        y = y.reshape((num_pts, 1))
        line.set_ydata(y)
        return line,
    animation = anm.FuncAnimation(fig, func=animate, frames=len(t), interval=interval, blit=True)
    animation.save(file_path)



class Domain:
    """
    Description:
        A class for implementing box domains
    Attrs:
        low: an array of floats specifying min in each dimension
        high: an array of floats specifying max in each dimension
    """
    def __init__(self, low, high):
        self.low = low 
        self.high = high
        self.dim = len(low)

    def sample(self, num_sample):
        return tf.random.uniform(shape=(num_sample, self.dim), minval=self.low, maxval=self.high)


class NNPlotter:
    def __init__(self, funcs, low, high, num_pts_per_dim=15):
        self.funcs = funcs
        self.domain = Domain(low, high)
        self.num_pts_per_dim = num_pts_per_dim
        self.coord_data = [tf.reshape(tf.linspace(low, self.domain.high[i], num=num_pts_per_dim), shape=(-1, 1)) for i, low in enumerate(self.domain.low)]

    def plot(self, file_path, t=None, style='standard', fig_size=(8, 8), solo=False, x_lim=None, y_lim=None, z_lim=None, wireframe=False):
        # determine the type of plot
        self.style = str(self.domain.dim) + 'd_' + style
        self.set_plot_fns(wireframe=wireframe)

        # take care of time dependency
        func_list = self.funcs if not solo else self.funcs[: 1]

        # generate data to plot
        geom_data = self.gen_geom_data(func_list, t)

        # generate arguments to feed plotting functions
        args, kwargs = self.gen_plot_args(geom_data)
        #print(args)

        # plot
        fig = plt.figure(figsize=fig_size)
        if self.style.startswith('1d') or self.style.endswith('heatmap'):
            ax = fig.add_subplot(111)
        else:
            ax = fig.add_subplot(111, projection='3d')
        for i, fn in enumerate(self.plot_fns):
            getattr(ax, fn)(*args[i], **kwargs[i])
        if t is not None:
            ax.set_title('time = {:.2f}'.format(t))
        self.set_plot_labels(ax, x_lim=x_lim, y_lim=y_lim, z_lim=z_lim)
        if self.style == '1d_standard':
            plt.legend()
        plt.savefig(file_path)

    def func_animate(self, file_path, t, num_frames=240, style='standard', fig_size=(8, 8), solo=False, x_lim=None, y_lim=None, z_lim=None, wireframe=False):
        # determine the type of plot
        self.style = str(self.domain.dim) + 'd_' + style
        self.set_plot_fns(wireframe=wireframe)
        func_list = self.funcs if not solo else self.funcs[: 1]
        fig = plt.figure(figsize=fig_size)
        if self.style.startswith('1d') or self.style.endswith('heatmap'):
            ax = fig.add_subplot(111)
        else:
            ax = fig.add_subplot(111, projection='3d')
        self.set_plot_labels(ax, x_lim=x_lim, y_lim=y_lim, z_lim=z_lim)

        def update_plot(frame):
            # take care of time dependency
            funcs = [lambda *args: func(tf.fill((self.num_pts_per_dim, 1), frame), *args) for func in func_list]
            # generate data to plot
            geom_data = self.gen_geom_data(funcs)
            # generate arguments to feed plotting functions
            args, kwargs = self.gen_plot_args(geom_data)
            for i, fn in enumerate(self.plot_fns):
                getattr(ax, fn)(*args[i], **kwargs[i])
            ax.set_title('time = {:.2f}'.format(frame))

            if self.style == '1d_standard':
                plt.legend()

        ani = anm.FuncAnimation(fig, update_plot, np.linspace(t[0], t[1], num=num_frames))
        ani.save(file_path, writer='ffmpeg', fps=24)

    @ut.timer
    def animate(self, file_path, t, num_frames=240, style='standard', fig_size=(8, 8), solo=False, x_lim=None, y_lim=None, z_lim=None, wireframe=False):
        # create folder to store images
        try:
            frames_folder = os.path.dirname(file_path) + '/{}'.format(os.path.basename(file_path).split('.')[0])
            os.mkdir(frames_folder)
        except:
            pass
        # determine the type of plot
        self.style = str(self.domain.dim) + 'd_' + style
        self.set_plot_fns(wireframe=wireframe)
        func_list = self.funcs if not solo else self.funcs[: 1]
        fig = plt.figure(figsize=fig_size)
        if self.style.startswith('1d') or self.style.endswith('heatmap'):
            ax = fig.add_subplot(111)
        else:
            ax = fig.add_subplot(111, projection='3d')

        @ut.timer
        def update_plot(frame, time_):
            ax.clear()

            # generate data to plot
            geom_data = self.gen_geom_data(func_list, time_)
            #print(geom_data)
            # generate arguments to feed plotting functions
            args, kwargs = self.gen_plot_args(geom_data)
            for i, fn in enumerate(self.plot_fns):
                getattr(ax, fn)(*args[i], **kwargs[i])
            ax.set_title('time = {:.2f}'.format(time_))
            self.set_plot_labels(ax, x_lim=x_lim, y_lim=y_lim, z_lim=z_lim)
            if self.style == '1d_standard':
                plt.legend()
            plt.savefig(frames_folder + '/frame_{}.png'.format(frame))
            print('Frame {} has been drawn.'.format(frame))

        for frame, time_ in enumerate(np.linspace(t[0], t[1], num=num_frames)):
            update_plot(frame, time_)

        height, width, layers = cv2.imread(frames_folder + '/frame_{}.png'.format(0)).shape
        video = cv2.VideoWriter(file_path, fourcc = cv2.VideoWriter_fourcc(*'mp4v'), frameSize=(width,height), fps=24)
        for frame in range(num_frames):
            video.write(cv2.imread(frames_folder + '/frame_{}.png'.format(frame)))
        cv2.destroyAllWindows()
        video.release()
        shutil.rmtree(frames_folder)

    def set_plot_fns(self, wireframe):
        if self.style.endswith('standard'):
            if self.style.startswith('1d'):
                self.plot_fns = ['plot', 'plot'][: len(self.funcs)]
            elif self.style.startswith('2d'):
                if wireframe:
                    self.plot_fns = ['plot_wireframe', 'plot_wireframe'][: len(self.funcs)]
                else:
                    self.plot_fns = ['plot_surface', 'plot_wireframe'][: len(self.funcs)]
        elif self.style.endswith('error'):
            if self.style.startswith('1d'):
                self.plot_fns = ['plot']
            elif self.style.startswith('2d'):
                self.plot_fns = ['plot_surface']
        elif self.style.endswith('heatmap'):
                self.plot_fns = ['imshow']

    def gen_geom_data(self, funcs, t = None):
        if self.style.endswith('standard'):
            if self.style.startswith('1d'):
                geom_data = []
                for func in funcs:
                    if t is None:
                        geom_data.append(tf.reshape(func(*self.coord_data), shape=(-1, )).numpy())
                    else:
                        t_ = tf.fill((self.num_pts_per_dim, 1), t)
                        geom_data.append(tf.reshape(func(t_, *self.coord_data), shape=(-1, )).numpy())
            elif self.style.startswith('2d'):
                geom_data = []
                for func in funcs:
                    data = np.empty((self.num_pts_per_dim, self.num_pts_per_dim))
                    for i in range(self.num_pts_per_dim):
                        y = self.coord_data[1]
                        x = tf.fill(y.shape, self.coord_data[0][i])
                        if t is None:
                            data[i, :] = tf.reshape(func(x, y), shape=(-1, )).numpy()
                        else:
                            t_ = tf.fill(x.shape, t)
                            data[i, :] = tf.reshape(func(t_, x, y), shape=(-1, )).numpy()
                    geom_data.append(data)


        elif self.style.endswith('error'):
            if self.style.startswith('1d'):
                if t is None:
                    geom_data = tf.reshape(funcs[0](*self.coord_data) - funcs[1](*self.coord_data), shape=(-1, )).numpy()
                else:
                    t_ = tf.fill((self.num_pts_per_dim, 1), t)
                    geom_data = tf.reshape(funcs[0](t_, *self.coord_data) - funcs[1](t_, *self.coord_data), shape=(-1, )).numpy()
            elif self.style.startswith('2d'):
                geom_data = np.empty((self.num_pts_per_dim, self.num_pts_per_dim))
                for i in range(self.num_pts_per_dim):
                    y = self.coord_data[1]
                    x = tf.fill(y.shape, self.coord_data[0][i])
                    if t is None:
                        geom_data[i, :] = tf.reshape(funcs[0](x, y) - funcs[1](x, y), shape=(-1, )).numpy()
                    else:
                        t_ = tf.fill(x.shape, t)
                        geom_data[i, :] = tf.reshape(funcs[0](t_, x, y) - funcs[1](t_, x, y), shape=(-1, )).numpy()

        elif self.style.endswith('heatmap'):
            geom_data = np.empty((self.num_pts_per_dim, self.num_pts_per_dim))
            for i in range(self.num_pts_per_dim):
                y = self.coord_data[1]
                x = tf.fill(y.shape, self.coord_data[0][i])
                if t is None:
                    geom_data[i, :] = tf.reshape(funcs[0](x, y) - funcs[1](x, y), shape=(-1, )).numpy()
                else:
                    t_ = tf.fill(x.shape, t)
                    geom_data[i, :] = tf.reshape(funcs[0](t_, x, y) - funcs[1](t_, x, y), shape=(-1, )).numpy()
        return geom_data

    def gen_plot_args(self, geom_data):
        args = []
        kwargs = []
        if self.style.endswith('standard'):
            if self.style.startswith('1d'):
                x_data = tf.reshape(self.coord_data[0], shape=(-1, )).numpy()
                for i, func in enumerate(self.funcs):
                    l, d = [], {}
                    l.append(x_data)
                    l.append(geom_data[i])
                    d['label'] = func.name
                    args.append(l)
                    kwargs.append(d)
            elif self.style.startswith('2d'):
                for i, func in enumerate(self.funcs):
                    d = {}
                    X, Y = np.meshgrid(*list(map(lambda x: tf.reshape(x, shape=(-1, )).numpy(), self.coord_data)))
                    d['X'] = X
                    d['Y'] = Y
                    d['Z'] = geom_data[i]
                    if i == 0:
                        d['cmap'] = 'viridis'
                    args.append([])
                    kwargs.append(d)

        elif self.style.endswith('error'):
            if self.style.startswith('1d'):
                l = []
                l.append(tf.reshape(self.coord_data[0], shape=(-1, )).numpy())
                l.append(geom_data)
                args.append(l)
                kwargs.append({})
            elif self.style.startswith('2d'):
                d = {}
                X, Y = np.meshgrid(*list(map(lambda x: tf.reshape(x, shape=(-1, )).numpy(), self.coord_data)))
                d['X'] = X
                d['Y'] = Y
                d['Z'] = geom_data
                d['cmap'] = 'viridis'
                args.append([])
                kwargs.append(d)

        elif self.style.endswith('heatmap'):
            d = {}
            d['X'] = geom_data
            d['cmap'] = 'viridis'
            d['extent'] = (self.domain.low[0], self.domain.high[0], self.domain.low[1], self.domain.high[1])
            d['origin'] = 'lower'
            args.append([])
            kwargs.append(d)
        return args, kwargs

    def set_plot_labels(self, ax, x_lim=None, y_lim=None, z_lim=None):
        if self.style.endswith('standard') or self.style.endswith('error'):
            if x_lim is not None:
                ax.set_xlim(x_lim)
            if y_lim is not None:
                ax.set_ylim(y_lim)
            if self.style.startswith('1d'):
                ax.set_xlabel('x')
            elif self.style.startswith('2d'):
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                if z_lim is not None:
                    ax.set_zlim(z_lim)

        elif self.style.endswith('error'):
            if x_lim is not None:
                ax.set_xlim(x_lim)
            if y_lim is not None:
                ax.set_ylim(y_lim)
            if self.style.startswith('1d'):
                ax.set_xlabel('x')
                ax.set_ylabel('error')
            if self.style.startswith('2d'):
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                if z_lim is not None:
                    ax.set_zlim(z_lim)

        elif self.style.endswith('heatmap'):
            ax.set_xlabel('x')
            ax.set_ylabel('y')
