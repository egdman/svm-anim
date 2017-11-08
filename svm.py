import os
import sys
import Tkinter as tk
from tkFont import Font as font
import json
import math
from collections import deque
from time import sleep
from itertools import izip, chain, repeat

root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root_dir, 'lib'))


class vec:
  def __init__(self, *comps):
    self.comps = tuple(float(comp) for comp in comps)

  def mul(self, scal):
    return vec(*(comp * scal for comp in self.comps))

  def normSq(self):
    return self.dot(self)

  def norm(self):
    return math.sqrt(self.normSq())

  def normalized(self):
    return self.mul(1. / self.norm())

  def plus(self, v):
    return vec(*(self_c + v_c for (self_c, v_c) in zip(self.comps, v.comps)))

  def minus(self, v):
    return self.plus(v.mul(-1))

  def dot(self, v):
    return sum(self_c*v_c for self_c, v_c in zip(self.comps, v.comps))

  def __getitem__(self, dim):
    return self.comps[dim]

  def __repr__(self):
    return self.comps.__repr__()

  def cross(u, v):
    return vec(*(
      u[1]*v[2] - u[2]*v[1],
      u[2]*v[0] - u[0]*v[2],
      u[0]*v[1] - u[1]*v[0]
    ))


def fit_svm(ptsNeg, ptsPos, w, bias, learnRateW, learnRateB, regParam, maxIters = 10000):
  totalNum = float(len(ptsPos) + len(ptsNeg))
  llambda = 2. / float(regParam * totalNum)
  print("lambda = {}".format(llambda))

  def labelsPos():
    return repeat( 1., len(ptsPos))

  def labelsNeg():
    return repeat(-1., len(ptsNeg))

  def data():
    return izip(chain(ptsNeg, ptsPos), chain(labelsNeg(), labelsPos()))

  def planeFunc(x):
    return w.dot(x) + bias

  def marginWidth():
    return 2. / w.normSq()

  def cost():
    return .5*llambda*w.normSq() +\
    sum((max(0., 1. - label * planeFunc(x)) for x, label in data())) / totalNum

  def isSupport(pt, label):
    return label * planeFunc(pt) < 1.0

  def lossGrad():
    gradW, gradB = vec(0, 0), 0
    for x, label in data():
      isSup = isSupport(x, label)
      gradW = gradW.plus( (x.mul(-label) if isSup else vec(0,0)).mul(learnRateW / totalNum) )
      gradB += (-label if isSup else 0) * learnRateB / totalNum
    return gradW, gradB

  for it in range(maxIters):
    lossGradW, gradB = lossGrad()
    gradW = w.mul(llambda * learnRateW).plus(lossGradW)

    if it % 100 == 0:
      print("cost = {} | grad norms: {}, {} | margin width: {}".format(
        cost(), gradW.norm(), abs(gradB), marginWidth()))

    # update learning rates
    if (it + 1) % int(maxIters / 10) == 0:
      learnRateW *= 0.5
      learnRateB *= 0.5
      print("lrW = {}, lrB = {}".format(learnRateW, learnRateB))

    # update w and bias
    w = w.minus(gradW)
    bias = bias - gradB
    yield w, bias


def get_bbox(pts):
  xl, yl =  float('inf'),  float('inf')
  xu, yu = -float('inf'), -float('inf')
  for pt in pts:
    xl = min(xl, pt[0])
    yl = min(yl, pt[1])
    xu = max(xu, pt[0])
    yu = max(yu, pt[1])
  return xl, xu, yl, yu


def normal_color(norm):
  r = int( 255. * (norm[0] + 1.)*.5 )
  g = int( 255. * (norm[1] + 1.)*.5 )
  b = 127
  return '#{:02x}{:02x}{:02x}'.format(r, g, b)


class Application(tk.Frame):
  def __init__(self, master=None):
    tk.Frame.__init__(self, master)
    self.grid()
    self.sz = 3
    self.line_color = '#5f5f5f'
    self.pos = []
    self.neg = []
    self.labels = []

    self.line_pts = []
    self.line = None

    self.line_tag = 'line_tag'
    self.line_pts_tag = 'line_pts_tag'
    self.supports_tag = 'support_vectors_tag'
    self.datapoints_tag = 'datapoints_tag'
    self.bbox_tag = 'bbox_tag'
    self.c_to_w_scale = 1.0
    self.w_to_c_scale = 1. / self.c_to_w_scale

    self.createWidgets()
    # self.create_data()


  def create_data(self):
    centerX = self.canvas_crds_of_origin[0]
    centerY = self.canvas_crds_of_origin[1]
    scale = .3 * self.canvas_size[1]
    self._add_point(centerX - 0.8*scale, centerY - 0.8*scale, 1)
    self._add_point(centerX - 0.5*scale, centerY - 0.8*scale, 1)
    self._add_point(centerX - 0.8*scale, centerY - 0.5*scale, 1)
    self._add_point(centerX - 0.5*scale, centerY - 0.5*scale, 1)

    self._add_point(centerX + 0.8*scale, centerY + 0.8*scale, -1)
    self._add_point(centerX + 0.5*scale, centerY + 0.8*scale, -1)
    self._add_point(centerX + 0.8*scale, centerY + 0.5*scale, -1)
    self._add_point(centerX + 0.5*scale, centerY + 0.5*scale, -1)




  def createWidgets(self):
    self.canvas = tk.Canvas(self, background='#000000', width=1400, height=1200,
        scrollregion=(0, 0, 1200, 900))

    self.canvas_size = (int(self.canvas["width"]), int(self.canvas["height"]))
    self.canvas_crds_of_origin = (.5 * self.canvas_size[0], .5 * self.canvas_size[1])

    # pack root window into OS window and make it fill the entire window
    self.pack(side = tk.LEFT, expand = True, fill = tk.BOTH)

    self.canvas.config(xscrollincrement=1, yscrollincrement=1)

    # bind right and left mouse clicks
    self.canvas.bind('<ButtonRelease-1>', self._left_up)
    self.canvas.bind('<ButtonRelease-3>', self._right_up)

    # bind ctrl-clicks
    self.canvas.bind('<Control-ButtonRelease-1>', self._ctrl_left_up)
    self.canvas.bind('<Control-ButtonRelease-3>', self._ctrl_right_up)

    self.canvas.bind('<KeyRelease-space>', self._update_line)
    # self.canvas.bind('<KeyPress>', self._debg)

    # bind mouse moved event
    # self.canvas.bind('<Motion>', self._mouse_moved)

    # draw origin cross
    ccoo = self.canvas_crds_of_origin
    csz = .25 * self.canvas_size[1]
    self.canvas.create_line(ccoo[0] + csz, ccoo[1], ccoo[0] - csz, ccoo[1], fill = '#585858')
    self.canvas.create_line(ccoo[0], ccoo[1] + csz, ccoo[0], ccoo[1] - csz, fill = '#585858')

    self.canvas.pack(side = tk.RIGHT, expand = True, fill = tk.BOTH)
    self.canvas.focus_set()


  
  def _get_event_modifiers(self, event):
    state = {
      'ctrl': event.state & 0x0004 != 0,
      'ralt': event.state & 0x0080 != 0,
      'lalt': event.state & 0x0008 != 0,
      'shift': event.state & 0x0001 != 0,
    }
    return set(modname for modname in state if state[modname])


  def _debg(self, event):
    print(event.keysym)
    print(event.type)
    print(self._get_event_modifiers(event))


  def _add_point(self, cx, cy, label):
    sz = self.sz
    color = '#aa1111' if label > 0 else '#11aa11'
    cloud = self.pos if label > 0 else self.neg
    cloud.append(vec( *self.canvas_to_world(cx, cy) ))
    self.draw_point(cloud[-1].comps, sz, color = color, tag = self.datapoints_tag)


  def _ctrl_left_up(self, event):
    self._add_point(event.x, event.y, 1)



  def _ctrl_right_up(self, event):
    self._add_point(event.x, event.y, -1)



  def _left_up(self, event):
    if len(self.line_pts) == 2:
      del self.line_pts[0]

    self.line_pts.append(vec( *self.canvas_to_world(event.x, event.y) ))
    if len(self.line_pts) == 2:
      self.line = self.get_line_coefs(*self.line_pts)
    self.redraw()



  def _right_up(self, event):
    x = vec(*self.canvas_to_world(event.x, event.y))
    print("x = {}, f(x) = {}".format(
      x.comps,
      self.line[0].dot(x) + self.line[1] if self.line is not None else "??"))



  def draw_point(self, world_crds, size, color, tag = None):
    ccrds = self.world_to_canvas(*world_crds)
    self.canvas.create_oval(
      ccrds[0] - size, ccrds[1] - size,
      ccrds[0] + size, ccrds[1] + size,
      fill = color, tag = tag)

  def draw_bbox(self, xl, xu, yl, yu):
    xl, yl = self.world_to_canvas(xl, yl)
    xu, yu = self.world_to_canvas(xu, yu)
    self.canvas.delete(self.bbox_tag)
    self.canvas.create_line(xl, yl, xu, yl, fill = self.line_color, tag = self.bbox_tag)
    self.canvas.create_line(xl, yl, xl, yu, fill = self.line_color, tag = self.bbox_tag)
    self.canvas.create_line(xu, yu, xu, yl, fill = self.line_color, tag = self.bbox_tag)
    self.canvas.create_line(xu, yu, xl, yu, fill = self.line_color, tag = self.bbox_tag)


  def _update_line(self, event):
    bbox = get_bbox(chain(self.pos, self.neg))
    problem_scale = max(abs(bbox[0] - bbox[1]), abs(bbox[2] - bbox[3]))
    print("scale = {}".format(problem_scale))
    self.draw_bbox(*bbox)

    if self.line is None: return

    def animate(svm):
      for _ in range(100):
        try:
          self.line = svm.next()
        except StopIteration:
          (w, bias) = self.line
          print("after: {}, {}".format(w.comps, bias))
          print("margin width = {}".format(2. / w.normSq()))
          return

      self.redraw_line()
      self.canvas.after(1, animate, svm)


    (w, bias) = self.line
    hyperplane_scale = 6. / problem_scale

    # scale hyperplane
    w = w.mul(hyperplane_scale)
    bias *= hyperplane_scale

    print("before: {}, {}".format(w.comps, bias))
    animate(fit_svm(self.neg, self.pos, w, bias,
      # learnRateW = 0.0005 / self.c_to_w_scale,     # scales as problem_scale^-1
      # learnRateB = 0.1,                            # does not scale
      # regParam = 0.0001 / (self.c_to_w_scale**2),  # scales as problem_scale^-2
      # maxIters = 10000

      learnRateW = 0.3 / problem_scale,     # scales as problem_scale^-1
      learnRateB = 0.1,                     # does not scale
      regParam = 36.0 / (problem_scale**2), # scales as problem_scale^-2
      maxIters = 20000
    ))

    # def isSupport(x, label):
    #   val = w.dot(x) + bias
    #   return val*label < 1.1

    # supports = []
    # for x in self.neg:
    #   if isSupport(x, -1):
    #     supports.append(x)
    # for x in self.pos:
    #   if isSupport(x, 1):
    #     supports.append(x)

    # self.canvas.delete(self.supports_tag)
    # for sv in supports:
    #   self.draw_point(sv.comps, 4, 'grey', self.supports_tag)


  def redraw_line(self):
    if self.line is None: return
    (w, bias) = self.line
    pt1, pt2 = self.get_points_on_line(w, bias, spacing = 10)
    pt3 = pt1.plus(w)
    pt1 = vec(*self.world_to_canvas(*pt1.comps))
    pt2 = vec(*self.world_to_canvas(*pt2.comps))
    pt3 = vec(*self.world_to_canvas(*pt3.comps))
    guide = pt2.minus(pt1)
    pFar1 = pt1.plus(guide.mul(200)).comps
    pFar2 = pt1.plus(guide.mul(-200)).comps

    self.canvas.delete(self.line_tag)
    self.canvas.create_line(pt1.comps[0], pt1.comps[1], pt3.comps[0], pt3.comps[1], fill = self.line_color, tag = self.line_tag)
    self.canvas.create_line(pFar1[0], pFar1[1], pFar2[0], pFar2[1], fill = 'white', tag = self.line_tag)



  def redraw(self):
    self.canvas.delete(self.line_pts_tag)

    if len(self.line_pts) > 0:
      self.draw_point(self.line_pts[0].comps, 2, 'yellow', self.line_pts_tag)
      if len(self.line_pts) > 1:
        self.draw_point(self.line_pts[1].comps, 4, 'cyan', self.line_pts_tag)

    self.redraw_line()



  def get_points_on_line(self, w, bias, spacing = 100):
    pt1 = w.mul(-bias / w.normSq())
    pt2 = pt1.plus(vec(-w.comps[1], w.comps[0]).normalized().mul(spacing))
    return pt1, pt2


  def get_line_coefs(self, u, v):
    R = v.minus(u).comps

    if abs(R[0]) > abs(R[1]):
      w = [0, 1]
      unknown_idx = 0
    else:
      w = [1, 0]
      unknown_idx = 1

    uidx = unknown_idx
    kidx = 1 - uidx # known idx
    w[uidx] = - float(R[kidx]) / float(R[uidx])
    w = vec(*w)
    # w = w.normalized()
    bias = - w.dot(u)
    print("w = {},  bias = {}".format(w.comps, bias))
    return w, bias


  def canvas_to_world(self, x, y):
    x -= self.canvas_crds_of_origin[0]
    y -= self.canvas_crds_of_origin[1]
    return x * self.c_to_w_scale, -y * self.c_to_w_scale

  def world_to_canvas(self, x, y):
    x *= self.w_to_c_scale
    y *= - self.w_to_c_scale
    return x + self.canvas_crds_of_origin[0], y + self.canvas_crds_of_origin[1]


class ProvideException(object):
    def __init__(self, func):
        self._func = func

    def __call__(self, *args):
        try:
            return self._func(*args)

        except StandardError, e:
          print('Exception was thrown: {}'.format(e))



@ProvideException
def main():
  app = Application()
  app.master.title('svm')
  app.mainloop() 


if __name__ == '__main__':
    main()
