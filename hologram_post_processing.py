# -*- coding: utf-8 -*-
"""
Created on Tue May 12 17:09:54 2020

@author: Connor
"""
import numpy as np
from skimage.restoration import unwrap_phase
from scipy.ndimage import rotate
from matplotlib.widgets import RectangleSelector, Button
from tkinter.filedialog import askopenfilename, asksaveasfilename
import h5py
import matplotlib.pyplot as plt
from matplotlib.path import Path
from pickle import dump 

class PostProcess:
    
    def __init__(self, data):
        self.data = data
        fig, ax = plt.subplots()
        ax.tick_params(axis='both', which='both', bottom=False, left=False, 
                       labelbottom=False, labelleft=False)
        plt.subplots_adjust(bottom=0.1, left=0.1, top=0.80, right=0.80)
        
        ax_crop = plt.axes([0.82, 0.7, 0.16, 0.06])
        ax_unwrap = plt.axes([0.82, 0.62, 0.16, 0.06])
        ax_wedge = plt.axes([0.82, 0.54, 0.16, 0.06])
        ax_leakage = plt.axes([0.82, 0.46, 0.16, 0.06])
        ax_reset = plt.axes([0.82, 0.38, 0.16, 0.06])
        ax_save = plt.axes([0.82, 0.3, 0.16, 0.06])
        
        self.b_crop = Button(ax_crop, 'Crop')
        self.b_crop.on_clicked(self.crop_active)
        self.b_unwrap = Button(ax_unwrap, 'Unwrap')
        self.b_unwrap.on_clicked(self.unwrap)
        self.b_wedge = Button(ax_wedge, 'Wedge')
        self.b_wedge.on_clicked(self.wedge_active)
        self.b_leakage = Button(ax_leakage, 'Vac Leakage')
        self.b_leakage.on_clicked(self.leak_active)
        self.b_reset = Button(ax_reset, 'Reset')
        self.b_reset.on_clicked(self.reset)
        self.b_save = Button(ax_save, 'Save')
        self.b_save.on_clicked(self.save)
        
        self.cid1 = fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.cid2 = fig.canvas.mpl_connect('button_release_event', self.mouse_release)
        self.cid3 = fig.canvas.mpl_connect('motion_notify_event', self.mouse_move)
        
        self.ind = 0
        self.x = [0, 0]
        self.y = [0, 0]
        self.theta = 0
        self.verts = []
        
        self.phase_calc()
        ax.imshow(self.phase)
        
        self.roi = RectangleSelector(ax, self.crop_select, drawtype='box', 
                                useblit=True, button=1, interactive=True)
        self.roi.active = False
        self.bkgnd = RectangleSelector(ax, self.bkgnd_select, drawtype='box', 
                                useblit=True, button=1, interactive=True)
        self.bkgnd.active = False
        self.fig, self.ax = fig, ax
        plt.show()
        
    def phase_calc(self):
        a = np.real(self.data)
        b = np.imag(self.data)
        amp = np.sqrt(a ** 2 + b ** 2)
        sin = b / amp
        sin_bool = sin > 0
        theta = np.zeros(amp.shape)
        
        for i in range(theta.shape[0]):
            for j in range(theta.shape[1]):
                if sin_bool[i,j]:
                    theta[i,j] = np.arccos(a[i,j] / amp[i,j])
                else:
                    theta[i,j] = 2 * np.pi - np.arccos(a[i,j] / amp[i,j])  
                    
        self.amp, self.phase_raw = amp, theta
        self.phase = theta
    
    def reset(self, event):
        self.phase = self.phase_raw
        self.roi.active = False
        self.bkgnd.active = False
        self.ind = 0
        self.ax.clear()
        self.ax.imshow(self.phase)
        self.fig.canvas.draw_idle()

    def unwrap(self, event):
        self.phase = unwrap_phase(self.phase)
        self.ax.clear()
        self.ax.imshow(self.phase)
        self.fig.canvas.draw_idle()
    
    def crop_select(self, eclick, erelease):
        self.roi.active = False
        self.ax.set_title('')
        img = self.phase
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        self.phase = img[int(y1):int(y2), int(x1):int(x2)]
        self.ax.clear()
        self.ax.imshow(self.phase)
        self.fig.canvas.draw_idle()
        
    def crop_active(self, event):
        self.roi.active = True
        self.bkgnd.active = False
        self.ax.set_title('Crop ROI:')
                 
    def bkgnd_select(self, eclick, erelease):
        self.bkgnd.active = False
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        grad = np.gradient(self.phase[int(y1):int(y2), int(x1):int(x2)])
        grad_x = np.average(grad[0]) * np.arange(self.phase.shape[0])
        grad_y = np.average(grad[1]) * np.arange(self.phase.shape[1])
        grad_mat = np.empty(self.phase.shape, dtype=float)
        for i in range(self.phase.shape[0]):
            for j in range(self.phase.shape[1]):
                grad_mat[i,j] = grad_x[i] + grad_y[j]
        self.phase = self.phase - grad_mat
        self.phase = self.phase - np.average(self.phase)
        self.grad_mat = grad_mat
        self.ax.clear()
        self.ax.imshow(self.phase)
        self.ax.set_title('')
        self.fig.canvas.draw_idle()
        
    def wedge_active(self, event):
        self.roi.active = False
        self.bkgnd.active = True
        self.ax.set_title('Choose vacuum background region:')
        self.fig.canvas.draw_idle()
        
    def leak_active(self, event):
        self.ax.set_title('Trace GB (drag mouse) from vacuum to bulk:')
        self.fig.canvas.draw_idle()
        self.ind = 1    
                
    def draw_line(self):
        self.line.set_data(self.x, self.y)
        self.fig.canvas.draw_idle()
    
    def mouse_move(self, event):
        if self.ind in [2, 3, 4]:
            self.x[1] = event.xdata
            self.y[1] = event.ydata
            self.draw_line()
    
    def mouse_release(self, event):
        if self.ind==2:
            self.ind = 0
            x1, x2, y1, y2 = self.x[0], self.x[1], self.y[0], self.y[1]
            dx = x2 - x1
            dy = y2 - y1
            # 5 pixel tolerance for rotating image along line
            if abs(dx) > 5 and abs(dy) > 5:
                theta = np.degrees(np.arctan(abs(dy) / abs(dx)))
                if dx < 0 and dy <= 0:
                    pass
                elif dx >= 0 and dy < 0:
                    theta = 180 - theta
                elif dx > 0 and dy >= 0:
                    theta += 180
                else:
                    theta = 360 - theta
                self.phase = rotate(self.phase, theta)
                self.ax.clear()
                self.ax.imshow(self.phase)
                self.ax.set_title('Trace sample edge (right-click to stop):')
                self.fig.canvas.draw_idle()
                self.ind = 3
        else:
            pass
            
    def onclick(self, event):
        # either draw line along GB (ind=1) or trace sample edge (ind=3)
        if event.button==1:
            if self.ind==1: # Draw line
                self.ind = 2
                self.x = [event.xdata, event.xdata]
                self.y = [event.ydata, event.ydata]
                self.line, = self.ax.plot(self.x, self.y, 'r', picker=5)
            elif self.ind==3:
                self.verts = np.array([[event.xdata, event.ydata]])
                self.ind = 4
                self.x = [event.xdata, event.xdata]
                self.y = [event.ydata, event.ydata]
                self.line, = self.ax.plot(self.x, self.y, 'r', picker=5)
            elif self.ind==4:
                self.verts = np.append(self.verts, [[event.xdata, event.ydata]], 
                                       axis=0)
                self.x = [event.xdata, event.xdata]
                self.y = [event.ydata, event.ydata]
                self.line, = self.ax.plot(self.x, self.y, 'r', picker=5)
        if event.button==3:
            if self.ind==4: # End edge trace
                self.ind = 0
                self.verts = np.append(self.verts, [[event.xdata, event.ydata]], 
                                       axis=0)
                self.ax.set_title('')
                self.vac_leakage() 
    
    def vac_leakage(self):
        phase = self.phase
        pts_temp = np.zeros(phase.shape, dtype=bool)
        if self.verts[0, 1] < self.verts[-1, 1]:
            verts_rest = np.array([[self.verts[-1, 0], phase.shape[0]], 
                               [phase.shape[1], phase.shape[0]], 
                               [phase.shape[1], 0], 
                               [self.verts[0, 0], 0],
                               [self.verts[0,0], self.verts[0,1]]])
            self.verts = np.append(self.verts, verts_rest, axis=0)
        else:
            verts_rest = np.array([[self.verts[-1, 0], 0], 
                               [phase.shape[1], 0], 
                               [phase.shape[1], phase.shape[0]], 
                               [self.verts[0, 0], phase.shape[0]],
                               [self.verts[0,0], self.verts[0,1]]])
            self.verts = np.append(self.verts, verts_rest, axis=0)
        jmin = np.min(self.verts[:, 0])
        jmax = np.max(self.verts[:, 0])
            
        path = Path(self.verts, closed=True)
        for j in range(pts_temp.shape[1]):
            if jmin <= j <= jmax:
                for i in range(pts_temp.shape[0]):
                    pts_temp[i, j] = path.contains_point([j, i])
                    
        phase_temp = pts_temp * phase
        avg = np.sum(phase_temp) / np.sum(phase_temp!=0)
        for j in range(phase.shape[1]):
            for i in range(phase.shape[0]):
                if phase[i, j]!=0:
                        phase[i, j] -= avg
        phase_temp = pts_temp * phase
        vac_field = np.trapz(phase_temp, axis=1)
        """
        Smooth vacuum correction
        vac_field_smooth = np.cumsum(vac_field, dtype=float)
        vac_field_smooth[11:] = vac_field_smooth[11:] - vac_field_smooth[:-11]
        vac_field_smooth = vac_field_smooth[10:] / 11
        smooth_ends = np.array([np.average(vac_field[:6]), np.average(vac_field[:7]),
                                np.average(vac_field[:8]), np.average(vac_field[:9]), 
                                np.average(vac_field[:10]), np.average(vac_field[-10:]),
                                np.average(vac_field[-9:]), np.average(vac_field[-8:]), 
                                np.average(vac_field[-7:]), np.average(vac_field[-6:])])
        vac_field = np.append(np.append(smooth_ends[:5], vac_field_smooth), 
                              smooth_ends[-5:])
        """
        vac_field = np.matmul(np.ones((phase.shape[1], 1), dtype=int), 
                              np.reshape(vac_field, (1, vac_field.size)))
        vac_field = np.transpose(vac_field)
        self.vac_field = vac_field
        self.phase_temp = phase_temp
        self.phase -= vac_field
        
        self.ax.clear()
        self.ax.imshow(self.phase)
        self.fig.canvas.draw_idle()
    
    def save(self, event):
        file_name = asksaveasfilename(title="Save output as:", defaultextension=".p", filetypes=[("", "*.p")])
        file = open(file_name, "wb")
        header = 'Pickle order 1:header 2:amp 3:phase 4:raw_amp 5:raw_phase'
        dump(header, file)
        dump(self.amp, file)
        dump(self.phase, file)
        dump(self.raw_amp, file)
        dump(self.raw_phase, file)
        file.close()
        
    @staticmethod
    def load():
        file_name = askopenfilename(filetypes=[("", "*.hdf5")])
        file = h5py.File(file_name, 'r')
        data = file['data'][()]
        return PostProcess(data)

def tilt_correct():
    pass        
    
def demo():
    file_name = askopenfilename(filetypes=[("", "*.hdf5")])
    file = h5py.File(file_name, 'r')
    data = file['data'][()]
    return PostProcess(data)
    
if __name__=='__main__':
    demo()