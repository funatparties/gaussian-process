# -*- coding: utf-8 -*-
"""@author: JoshM"""

import wx
import gaussianprocess as gp

class TestFrame(wx.Frame):
    def __init__(self):
        super().__init__(parent=None, title="Test Frame")
        self.panel = KernelRadioBox(self)
        self.Show()

class KernelRadioBox(wx.Panel):
    kernels = [gp.Linear(), gp.SquaredExponential(), gp.Periodic(),
                    gp.Polynomial()]
    
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        
        kernel_names = [k.name for k in self.kernels]
        
        self.mainsizer = wx.BoxSizer(wx.VERTICAL)
        self.rb = wx.RadioBox(
                self, -1, "Test title rb", wx.DefaultPosition, wx.DefaultSize,
                kernel_names, 1, wx.RA_SPECIFY_COLS
                )
        self.mainsizer.Add(self.rb, 0, wx.ALL, 20)
        self.textsizer = wx.FlexGridSizer(2,5,5)
        self.mainsizer.Add(self.textsizer, 0, wx.ALIGN_LEFT | wx.ALL, 5)
        

        self.SetSizer(self.mainsizer)
        
        self.Bind(wx.EVT_RADIOBOX, self.EvtRadioBox, self.rb)
        
    def EvtRadioBox(self, event):
        self.textsizer.Clear(True)
        self.labels = []
        self.fields = []
        config = self.kernels[self.rb.GetSelection()].config
        if len(config):
            for k,v in config.items():
                st = wx.StaticText(self, -1, k, style=wx.ALIGN_CENTER_HORIZONTAL)
                txtctrl = wx.TextCtrl(self, -1, str(v), size=wx.Size(50,20))
                self.textsizer.Add(st,0, wx.ALIGN_CENTER_HORIZONTAL)
                self.labels.append(k)
                self.textsizer.Add(txtctrl,0, wx.ALIGN_LEFT)
                self.fields.append(txtctrl)
            b2 = wx.Button(self, -1, "Default")
            self.Bind(wx.EVT_BUTTON, self.EvtDefaultButton, b2)
            self.textsizer.Add(b2,0,wx.ALL | wx.ALIGN_CENTER_HORIZONTAL,5)
            b1 = wx.Button(self, -1, "Apply")
            self.Bind(wx.EVT_BUTTON, self.EvtApplyButton, b1)
            self.textsizer.Add(b1,0,wx.ALL | wx.ALIGN_CENTER_HORIZONTAL,5)
            
            
        self.mainsizer.Layout()
        
    def EvtApplyButton(self, event):
        field_data = [t.GetLineText(0) for t in self.fields]
        try:
            field_data = [self.convert_input(s) for s in field_data]
        except ValueError:
            print("Please enter a valid number.")
            #TODO: display error message
        #TODO: validate input
        config = {k:v for k,v in zip(self.labels,field_data)}
        self.kernels[self.rb.GetSelection()].config = config
    
    def EvtDefaultButton(self, event):
        pass
    
    def convert_input(self,s):
        s = s.strip()
        s = float(s)
        return s
        
if __name__ == '__main__':
    app = wx.App(False)
    frame = TestFrame()
    app.MainLoop()