import os
import sys
import Tkinter as tk
from tkFont import Font as font
import json
import math
from collections import deque
from time import sleep
from itertools import izip, chain, repeat
from operator import itemgetter

class vec(object):
    """
    vector of arbitrary size
    """
    def __init__(self, *comps):
        self.comps = tuple(float(c) for c in comps)

    def dot(self, right):
        return sum((c0 * c1 for (c0, c1) in izip(self.comps, right.comps)))

    def normSq(self):
        return self.dot(self)

    def norm(self):
        return math.sqrt(self.normSq())

    # multiply by a scalar on the right
    def __mul__(self, right_scalar):
        return vec(*(c * right_scalar for c in self.comps))

    # multiply by a scalar on the left
    def __rmul__(self, left_scalar):
        return vec(*(left_scalar * c for c in self.comps))

    # negate
    def __neg__(self):
        return -1. * self

    # add vector
    def __add__(self, right):
        return vec(*(c0 + c1 for (c0, c1) in izip(self.comps, right.comps)))

    # subtract vector
    def __sub__(self, right):
        return self + (-right)

    # scalar division
    def __div__(self, right_scalar):
        return (1. / right_scalar) * self

    # [] getter
    def __getitem__(self, key):
        return self.comps[key]

    # equality test
    def __eq__(self, right):
        return self.comps == right.comps

    # inequality test
    def __ne__(self, right):
        return not self == right

    # hashing support
    def __hash__(self):
        return hash(self.comps)

    def __len__(self):
        return len(self.comps)

    def __repr__(self):
        return self.comps.__repr__()

    def normalized(self):
        return self / self.norm()

    def append(self, *tail):
        return vec(*chain(self.comps, tail))

    def prepend(self, *head):
        return vec(*chain(head, self.comps))


    @staticmethod
    def aabb(*points):
        """
        returns min and max corners of the axis-aligned bounding box of points
        """
        ndim = len(points[0].comps)
        min_corner, max_corner = [], []
        for dim in range(ndim):
            min_corner.append(min(points, key = itemgetter(dim)) [dim])
            max_corner.append(max(points, key = itemgetter(dim)) [dim])
        return (vec(*min_corner), vec(*max_corner))

    
    @staticmethod
    def cross3(u, v):
        """
        returns 3d vector
        requires at least 3d vectors
        """
        return vec(
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0])


    @staticmethod
    def cross2(u, v):
        """
        returns scalar
        requires at least 2d vectors
        """
        return u[0] * v[1] - u[1] * v[0]




def fit_svm(ptsNeg, ptsPos, w, bias, learnRateW, learnRateB, regParam, maxIters = 10000):
  totalNum = float(len(ptsPos) + len(ptsNeg))
  llambda = 2. / float(regParam * totalNum)
  print("lambda = {}".format(llambda))
  print("learnRateW = {}".format(learnRateW))
  print("learnRateB = {}".format(learnRateB))

  def labelsPos():
    return repeat( 1., len(ptsPos))

  def labelsNeg():
    return repeat(-1., len(ptsNeg))

  def data():
    return izip(chain(ptsNeg, ptsPos), chain(labelsNeg(), labelsPos()))

  def planeFunc(x):
    return w.dot(x) + bias

  def cost():
    return .5*llambda*w.normSq() +\
    sum((max(0., 1. - label * planeFunc(x)) for x, label in data())) / totalNum

  def isSupport(pt, label):
    return label * planeFunc(pt) < 1.0


  def lossGrad():
    gradW, gradB = vec(0, 0), 0
    multiplier = 1. / totalNum
    for x, label in data():
      isSup = isSupport(x, label)
      gradW += multiplier * (-label * x if isSup else vec(0,0))
      gradB += multiplier * (-label if isSup else 0)
    return gradW, gradB


  for it in range(maxIters):
    lossGradW, gradB = lossGrad()
    gradW = llambda * w + lossGradW

    stepW = gradW * learnRateW
    stepB = gradB * learnRateB

    if it % int(maxIters / 10) == 0:
      print("cost = {} | step: {}, {} | margin width: {}".format(
        cost(), stepW.norm(), stepB, 2. / w.normSq()))

    # update learning rates
    if (it + 1) % int(maxIters / 10) == 0:
      learnRateW *= 0.9
      learnRateB *= 0.9
      print("lrW = {}, lrB = {}".format(learnRateW, learnRateB))

    # update w and bias
    w -= stepW
    bias -= stepB
    yield w, bias


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

    self.line_tag = 'line_tag'
    self.line_pts_tag = 'line_pts_tag'
    self.supports_tag = 'support_vectors_tag'
    self.datapoints_tag = 'datapoints_tag'
    self.bbox_tag = 'bbox_tag'

    self.worldToCanvas = 500.
    self.canvasToWorld = 1. / self.worldToCanvas

    self.createWidgets()
    self.create_data()

    self.line_pts = []
    self.line = None

    #### EXPERIMENTAL ####
    self.line_pts = [vec(-1, 0), vec(1, 0)]
    self.line = self.get_line_coefs(*self.line_pts)
    self.redraw()
    #### #### #### #### ####



  def create_data(self):
    centerX = self.canvas_crds_of_origin[0]
    centerY = self.canvas_crds_of_origin[1]
    worldScale = 2.0
    scale = worldScale * self.worldToCanvas
    self._add_point(centerX - .5 * scale, centerY - .5 * scale, 1)
    self._add_point(centerX - .25 * scale, centerY - .5 * scale, 1)
    self._add_point(centerX - .5 * scale, centerY - .25 * scale, 1)
    self._add_point(centerX - .25 * scale, centerY - .25 * scale, 1)

    self._add_point(centerX + .5 * scale, centerY + .5 * scale, -1)
    self._add_point(centerX + .25 * scale, centerY + .5 * scale, -1)
    self._add_point(centerX + .5 * scale, centerY + .25 * scale, -1)
    self._add_point(centerX + .25 * scale, centerY + .25 * scale, -1)




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
    chart = (
      ('ctrl', 0x0004),
      ('ralt', 0x0080),
      ('lalt', 0x0008),
      ('shift', 0x0001),
    )
    return set(modname for (modname, flag) in chart if event.state & flag != 0)


  def _debg(self, event):
    print(event.keysym)
    print(event.type)
    print(self._get_event_modifiers(event))


  def _add_point(self, cx, cy, label):
    color = '#aa1111' if label > 0 else '#11aa11'
    categ = self.pos if label > 0 else self.neg
    worldLocation = vec(*self.canvas_to_world(cx, cy))
    print(worldLocation)

    categ.append(worldLocation)
    self.draw_point(categ[-1].comps, self.sz, color = color, tag = self.datapoints_tag)


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
      x, self.line[0].dot(x) + self.line[1] if self.line is not None else "??"))



  def draw_point(self, world_crds, size, color, tag = None):
    ccrds = self.world_to_canvas(*world_crds)
    self.canvas.create_oval(
      ccrds[0] - size, ccrds[1] - size,
      ccrds[0] + size, ccrds[1] + size,
      fill = color, tag = tag)

  def draw_box(self, xl, xu, yl, yu, canvas_tag):
    xl, yl = self.world_to_canvas(xl, yl)
    xu, yu = self.world_to_canvas(xu, yu)
    self.canvas.delete(self.bbox_tag)
    self.canvas.create_line(xl, yl, xu, yl, fill = self.line_color, tag = canvas_tag)
    self.canvas.create_line(xl, yl, xl, yu, fill = self.line_color, tag = canvas_tag)
    self.canvas.create_line(xu, yu, xu, yl, fill = self.line_color, tag = canvas_tag)
    self.canvas.create_line(xu, yu, xl, yu, fill = self.line_color, tag = canvas_tag)


  def _update_line(self, event):
    if self.line is None: return

    def animate(svm):
      for _ in range(100):
        try:
          self.line = svm.next()
        except StopIteration:
          (w, bias) = self.line
          print("after: {}, {}".format(w, bias))
          print("margin width = {}".format(2. / w.normSq()))
          return

      self.redraw_line()
      self.canvas.after(1, animate, svm) # schedule next call


    # find problem scale
    bbox_min, bbox_max = vec.aabb(*chain(self.pos, self.neg))
    bbox_diag = bbox_max - bbox_min
    problem_scale = max(bbox_diag[0], bbox_diag[1])
    print("problem_scale = {}".format(problem_scale))
    self.draw_box(bbox_min[0], bbox_max[0], bbox_min[1], bbox_max[1], self.bbox_tag)

    (w, bias) = self.line
    # # scale hyperplane
    planeMult = 1. / (w.norm() * problem_scale)
    w *= planeMult
    bias *= planeMult
    self.line = (w, bias)
    self.redraw()

    print("before: {}, {}".format(w, bias))


    animate(fit_svm(self.neg, self.pos, w, bias,
      learnRateW = 0.5 / (problem_scale),
      learnRateB = 0.1,
      regParam = 100 / (problem_scale**2),
      maxIters = 20000
    ))


  def redraw_line(self):
    if self.line is None: return
    (w, bias) = self.line
    pt1, pt2 = self.get_points_on_line(w, bias, spacing = 10)
    pt3 = pt1 + w
    pt1 = vec(*self.world_to_canvas(*pt1.comps))
    pt2 = vec(*self.world_to_canvas(*pt2.comps))
    pt3 = vec(*self.world_to_canvas(*pt3.comps))
    guide = pt2 - pt1
    pFar1 = pt1 + 200 * guide
    pFar2 = pt1 - 200 * guide

    self.canvas.delete(self.line_tag)
    self.canvas.create_line(pt1[0], pt1[1], pt3[0], pt3[1], fill = self.line_color, tag = self.line_tag)
    self.canvas.create_line(pFar1[0], pFar1[1], pFar2[0], pFar2[1], fill = 'white', tag = self.line_tag)



  def redraw(self):
    self.canvas.delete(self.line_pts_tag)

    if len(self.line_pts) > 0:
      self.draw_point(self.line_pts[0].comps, 2, 'yellow', self.line_pts_tag)
      if len(self.line_pts) > 1:
        self.draw_point(self.line_pts[1].comps, 4, 'cyan', self.line_pts_tag)

    self.redraw_line()



  def get_points_on_line(self, w, bias, spacing = 100):
    pt1 = -bias * w / w.normSq()
    pt2 = pt1 + spacing * (vec(-w[1], w[0]).normalized())
    return pt1, pt2


  def get_line_coefs(self, u, v):
    R = (v - u).comps

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
    bias = - w.dot(u)
    # print("w = {},  bias = {}".format(w, bias))
    return w, bias


  def canvas_to_world(self, x, y):
    x -= self.canvas_crds_of_origin[0]
    y -= self.canvas_crds_of_origin[1]
    return x * self.canvasToWorld, -y * self.canvasToWorld

  def world_to_canvas(self, x, y):
    x *= self.worldToCanvas
    y *= - self.worldToCanvas
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
