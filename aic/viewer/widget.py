#!/usr/bin/env python3.9
# *-* coding: utf-8*-*
"""
Copyright (C) 2022 Project AIC - All Rights Reserved.

Unauthorized copy of this file, via any medium is strictly
prohibited. Proprietary and confidential.

Note
----
Library for the Widget add-ons.
"""

import os

import numpy as np
import vtk
from vedo import Assembly
from vedo import getColor
from vedo import precision
from vedo import settings
from vedo import shapes


class Button(object):
    """Build a Button object to be shown in the rendering window."""

    def __init__(
        self, fnc, states, c, bc, pos, size, font, bold, italic, alpha, angle
    ):
        """Init function."""
        self._status = 0
        self.states = states
        self.colors = c
        self.bcolors = bc
        self.function = fnc
        self.actor = vtk.vtkTextActor()
        self.actor.GetActualPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        self.actor.SetPosition(pos[0], pos[1])
        self.framewidth = 2
        self.offset = 5
        self.spacer = " "

        self.textproperty = self.actor.GetTextProperty()
        self.textproperty.SetJustificationToCentered()
        if font.lower() == "courier":
            self.textproperty.SetFontFamilyToCourier()
        elif font.lower() == "times":
            self.textproperty.SetFontFamilyToTimes()
        else:
            self.textproperty.SetFontFamilyToArial()
        self.textproperty.SetFontSize(size)
        self.textproperty.SetBackgroundOpacity(alpha)
        self.textproperty.BoldOff()
        if bold:
            self.textproperty.BoldOn()
        self.textproperty.ItalicOff()
        if italic:
            self.textproperty.ItalicOn()
        self.textproperty.ShadowOff()
        self.textproperty.SetOrientation(angle)
        self.showframe = hasattr(self.textproperty, "FrameOn")
        self.status(0)

    def status(self, s=None):
        """Set/Get the status of the button."""
        if s is None:
            return self.states[self._status]

        if isinstance(s, str):
            s = self.states.index(s)
        self._status = s
        self.textproperty.SetLineOffset(self.offset)
        self.actor.SetInput(self.spacer + self.states[s] + self.spacer)
        s = s % len(self.colors)  # to avoid mismatch
        self.textproperty.SetColor(getColor(self.colors[s]))
        bcc = np.array(getColor(self.bcolors[s]))
        self.textproperty.SetBackgroundColor(bcc)
        if self.showframe:
            self.textproperty.FrameOn()
            self.textproperty.SetFrameWidth(self.framewidth)
            self.textproperty.SetFrameColor(np.sqrt(bcc))
        return self

    def switch(self):
        """Change button status to the next defined status in states list."""
        self._status = (self._status + 1) % len(self.states)
        self.status(self._status)
        return self


class Axes(object):
    """Class using axis attributes."""

    def __init__(self, interactor, lc=(1, 1, 1)):
        """Init function."""
        self.ax = vtk.vtkAxesActor()
        self.interactor = interactor
        self.widget = vtk.vtkOrientationMarkerWidget()
        self.properties(lc)
        self.addIcon()

    def properties(self, lc):
        """Define properties."""
        self.ax.SetShaftTypeToCylinder()
        self.ax.SetCylinderRadius(0.03)
        self.ax.SetXAxisLabelText("x")
        self.ax.SetYAxisLabelText("y")
        self.ax.SetZAxisLabelText("z")
        self.ax.GetXAxisShaftProperty().SetColor(1, 0, 0)
        self.ax.GetYAxisShaftProperty().SetColor(0, 1, 0)
        self.ax.GetZAxisShaftProperty().SetColor(0, 0, 1)
        self.ax.GetXAxisTipProperty().SetColor(1, 0, 0)
        self.ax.GetYAxisTipProperty().SetColor(0, 1, 0)
        self.ax.GetZAxisTipProperty().SetColor(0, 0, 1)
        self.ax.GetXAxisCaptionActor2D().GetCaptionTextProperty().BoldOff()
        self.ax.GetYAxisCaptionActor2D().GetCaptionTextProperty().BoldOff()
        self.ax.GetZAxisCaptionActor2D().GetCaptionTextProperty().BoldOff()
        self.ax.GetXAxisCaptionActor2D().GetCaptionTextProperty().ItalicOff()
        self.ax.GetYAxisCaptionActor2D().GetCaptionTextProperty().ItalicOff()
        self.ax.GetZAxisCaptionActor2D().GetCaptionTextProperty().ItalicOff()
        self.ax.GetXAxisCaptionActor2D().GetCaptionTextProperty().ShadowOff()
        self.ax.GetYAxisCaptionActor2D().GetCaptionTextProperty().ShadowOff()
        self.ax.GetZAxisCaptionActor2D().GetCaptionTextProperty().ShadowOff()
        self.ax.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetColor(lc)
        self.ax.GetYAxisCaptionActor2D().GetCaptionTextProperty().SetColor(lc)
        self.ax.GetZAxisCaptionActor2D().GetCaptionTextProperty().SetColor(lc)
        self.ax.PickableOff()

    def addIcon(self, size=0.1):
        """Add Icon values."""
        self.widget.SetOrientationMarker(self.ax)
        self.widget.SetInteractor(self.interactor)
        self.widget.SetViewport(0, 0, size * 2, size * 2)
        self.widget.EnabledOn()
        self.widget.InteractiveOff()


class Grid(object):
    """Class using the Grid attributes."""

    def __init__(self, actor):
        """Init function."""
        self.grid = None
        # will cause popping the defaults
        self.axes = dict()
        self.axes_copy = 1
        self.actor = actor
        self.properties()

    def properties(self, c=(0.9, 0.9, 0.9)):
        """Set properties."""
        xtitle = self.axes.pop("xtitle", "x")
        ytitle = self.axes.pop("ytitle", "y")
        ztitle = self.axes.pop("ztitle", "z")

        limitRatio = self.axes.pop("limitRatio", 20)
        vbb, sizes, min_bns, max_bns = self.computeVisibleBounds(self.actor)
        if sizes[0] and (
            sizes[1] / sizes[0] > limitRatio
            or sizes[2] / sizes[0] > limitRatio
        ):
            sizes[0] = 0
            xtitle = ""
        if sizes[1] and (
            sizes[0] / sizes[1] > limitRatio
            or sizes[2] / sizes[1] > limitRatio
        ):
            sizes[1] = 0
            ytitle = ""
        if sizes[2] and (
            sizes[0] / sizes[2] > limitRatio
            or sizes[1] / sizes[2] > limitRatio
        ):
            sizes[2] = 0
            ztitle = ""
        rats = []
        if sizes[0]:
            rats += [sizes[1] / sizes[0], sizes[2] / sizes[0]]
        if sizes[1]:
            rats += [sizes[0] / sizes[1], sizes[2] / sizes[1]]
        if sizes[2]:
            rats += [sizes[0] / sizes[2], sizes[1] / sizes[2]]
        if not len(rats):
            return
        rats = max(rats)
        if rats == 0:
            return

        nrDiv = max(1, int(6.5 / rats))
        numberOfDivisions = self.axes.pop("numberOfDivisions", nrDiv)
        axesLineWidth = self.axes.pop("axesLineWidth", 1)
        gridLineWidth = self.axes.pop("gridLineWidth", 1)
        reorientShortTitle = self.axes.pop("reorientShortTitle", True)
        originMarkerSize = self.axes.pop("originMarkerSize", 0)
        enableLastLabel = self.axes.pop("enableLastLabel", False)
        titleDepth = self.axes.pop("titleDepth", 0)

        xTitlePosition = self.axes.pop("xTitlePosition", 0.95)
        yTitlePosition = self.axes.pop("yTitlePosition", 0.95)
        zTitlePosition = self.axes.pop("zTitlePosition", 0.95)

        xTitleOffset = self.axes.pop("xTitleOffset", 0.05)
        yTitleOffset = self.axes.pop("yTitleOffset", 0.05)
        zTitleOffset = self.axes.pop("zTitleOffset", 0.05)

        xTitleJustify = self.axes.pop("xTitleJustify", "top-right")
        yTitleJustify = self.axes.pop("yTitleJustify", "bottom-right")
        zTitleJustify = self.axes.pop("zTitleJustify", "bottom-right")

        xTitleRotation = self.axes.pop("xTitleRotation", 0)
        yTitleRotation = self.axes.pop("yTitleRotation", 90)
        zTitleRotation = self.axes.pop("zTitleRotation", 135)

        xTitleSize = self.axes.pop("xTitleSize", 0.025)
        yTitleSize = self.axes.pop("yTitleSize", 0.025)
        zTitleSize = self.axes.pop("zTitleSize", 0.025)

        xTitleColor = self.axes.pop("xTitleColor", c)
        yTitleColor = self.axes.pop("yTitleColor", c)
        zTitleColor = self.axes.pop("zTitleColor", c)

        xKeepAspectRatio = self.axes.pop("xKeepAspectRatio", True)
        yKeepAspectRatio = self.axes.pop("yKeepAspectRatio", True)
        zKeepAspectRatio = self.axes.pop("zKeepAspectRatio", True)

        xyGrid = self.axes.pop("xyGrid", True)
        yzGrid = self.axes.pop("yzGrid", True)
        zxGrid = self.axes.pop("zxGrid", False)
        xyGrid2 = self.axes.pop("xyGrid2", False)
        yzGrid2 = self.axes.pop("yzGrid2", False)
        zxGrid2 = self.axes.pop("zxGrid2", False)

        xyGridTransparent = self.axes.pop("xyGridTransparent", False)
        yzGridTransparent = self.axes.pop("yzGridTransparent", False)
        zxGridTransparent = self.axes.pop("zxGridTransparent", False)
        xyGrid2Transparent = self.axes.pop("xyGrid2Transparent", False)
        yzGrid2Transparent = self.axes.pop("yzGrid2Transparent", False)
        zxGrid2Transparent = self.axes.pop("zxGrid2Transparent", False)

        xyPlaneColor = self.axes.pop("xyPlaneColor", c)
        yzPlaneColor = self.axes.pop("yzPlaneColor", c)
        zxPlaneColor = self.axes.pop("zxPlaneColor", c)
        xyGridColor = self.axes.pop("xyGridColor", c)
        yzGridColor = self.axes.pop("yzGridColor", c)
        zxGridColor = self.axes.pop("zxGridColor", c)
        xyAlpha = self.axes.pop("xyAlpha", 0.05)
        yzAlpha = self.axes.pop("yzAlpha", 0.05)
        zxAlpha = self.axes.pop("zxAlpha", 0.05)

        xLineColor = self.axes.pop("xLineColor", c)
        yLineColor = self.axes.pop("yLineColor", c)
        zLineColor = self.axes.pop("zLineColor", c)

        xHighlightZero = self.axes.pop("xHighlightZero", False)
        yHighlightZero = self.axes.pop("yHighlightZero", False)
        zHighlightZero = self.axes.pop("zHighlightZero", False)
        xHighlightZeroColor = self.axes.pop("xHighlightZeroColor", "red")
        yHighlightZeroColor = self.axes.pop("yHighlightZeroColor", "green")
        zHighlightZeroColor = self.axes.pop("zHighlightZeroColor", "blue")

        showTicks = self.axes.pop("showTicks", True)
        xTickRadius = self.axes.pop("xTickRadius", 0.005)
        yTickRadius = self.axes.pop("yTickRadius", 0.005)
        zTickRadius = self.axes.pop("zTickRadius", 0.005)

        xTickThickness = self.axes.pop("xTickThickness", 0.0025)
        yTickThickness = self.axes.pop("yTickThickness", 0.0025)
        zTickThickness = self.axes.pop("zTickThickness", 0.0025)

        xTickColor = self.axes.pop("xTickColor", xLineColor)
        yTickColor = self.axes.pop("yTickColor", yLineColor)
        zTickColor = self.axes.pop("zTickColor", zLineColor)

        xMinorTicks = self.axes.pop("xMinorTicks", 1)
        yMinorTicks = self.axes.pop("yMinorTicks", 1)
        zMinorTicks = self.axes.pop("zMinorTicks", 1)

        tipSize = self.axes.pop("tipSize", 0.01)

        xLabelPrecision = self.axes.pop("xLabelPrecision", 2)
        yLabelPrecision = self.axes.pop("yLabelPrecision", 2)
        zLabelPrecision = self.axes.pop("zLabelPrecision", 2)

        xLabelSize = self.axes.pop("xLabelSize", 0.0175)
        yLabelSize = self.axes.pop("yLabelSize", 0.0175)
        zLabelSize = self.axes.pop("zLabelSize", 0.0175)

        xLabelOffset = self.axes.pop("xLabelOffset", 0.015)
        yLabelOffset = self.axes.pop("yLabelOffset", 0.015)
        zLabelOffset = self.axes.pop("zLabelOffset", 0.01)

        ########################
        step = np.min(sizes[np.nonzero(sizes)]) / numberOfDivisions
        rx, ry, rz = np.rint(sizes / step).astype(int)
        if rx == 0:
            xtitle = ""
        if ry == 0:
            ytitle = ""
        if rz == 0:
            ztitle = ""

        if enableLastLabel:
            enableLastLabel = 1
        else:
            enableLastLabel = 0

        lines = []
        if xtitle:
            lines.append(
                shapes.Line(
                    [0, 0, 0], [1, 0, 0], c=xLineColor, lw=axesLineWidth
                )
            )
        if ytitle:
            lines.append(
                shapes.Line(
                    [0, 0, 0], [0, 1, 0], c=yLineColor, lw=axesLineWidth
                )
            )
        if ztitle:
            lines.append(
                shapes.Line(
                    [0, 0, 0], [0, 0, 1], c=zLineColor, lw=axesLineWidth
                )
            )

        grids = []
        if xyGrid and xtitle and ytitle:
            gxy = shapes.Grid(
                pos=(0.5, 0.5, 0), normal=[0, 0, 1], resx=rx, resy=ry
            )
            gxy.alpha(xyAlpha).wireframe(xyGridTransparent).c(xyPlaneColor).lw(
                gridLineWidth
            ).lc(xyGridColor)
            grids.append(gxy)
        if yzGrid and ytitle and ztitle:
            gyz = shapes.Grid(
                pos=(0, 0.5, 0.5), normal=[1, 0, 0], resx=rz, resy=ry
            )
            gyz.alpha(yzAlpha).wireframe(yzGridTransparent).c(yzPlaneColor).lw(
                gridLineWidth
            ).lc(yzGridColor)
            grids.append(gyz)
        if zxGrid and ztitle and xtitle:
            gzx = shapes.Grid(
                pos=(0.5, 0, 0.5), normal=[0, 1, 0], resx=rz, resy=rx
            )
            gzx.alpha(zxAlpha).wireframe(zxGridTransparent).c(zxPlaneColor).lw(
                gridLineWidth
            ).lc(zxGridColor)
            grids.append(gzx)

        grids2 = []
        if xyGrid2 and xtitle and ytitle:
            gxy2 = shapes.Grid(
                pos=(0.5, 0.5, 1), normal=[0, 0, 1], resx=rx, resy=ry
            )
            gxy2.alpha(xyAlpha).wireframe(xyGrid2Transparent).c(
                xyPlaneColor
            ).lw(gridLineWidth).lc(xyGridColor)
            grids2.append(gxy2)
        if yzGrid2 and ytitle and ztitle:
            gyz2 = shapes.Grid(
                pos=(1, 0.5, 0.5), normal=[1, 0, 0], resx=rz, resy=ry
            )
            gyz2.alpha(yzAlpha).wireframe(yzGrid2Transparent).c(
                yzPlaneColor
            ).lw(gridLineWidth).lc(yzGridColor)
            grids2.append(gyz2)
        if zxGrid2 and ztitle and xtitle:
            gzx2 = shapes.Grid(
                pos=(0.5, 1, 0.5), normal=[0, 1, 0], resx=rz, resy=rx
            )
            gzx2.alpha(zxAlpha).wireframe(zxGrid2Transparent).c(
                zxPlaneColor
            ).lw(gridLineWidth).lc(zxGridColor)
            grids2.append(gzx2)

        highlights = []
        if xyGrid and xtitle and ytitle:
            if xHighlightZero and min_bns[0] <= 0 and max_bns[1] > 0:
                xhl = -min_bns[0] / sizes[0]
                hxy = shapes.Line(
                    [xhl, 0, 0], [xhl, 1, 0], c=xHighlightZeroColor
                )
                hxy.alpha(np.sqrt(xyAlpha)).lw(gridLineWidth * 2)
                highlights.append(hxy)
            if yHighlightZero and min_bns[2] <= 0 and max_bns[3] > 0:
                yhl = -min_bns[2] / sizes[1]
                hyx = shapes.Line(
                    [0, yhl, 0], [1, yhl, 0], c=yHighlightZeroColor
                )
                hyx.alpha(np.sqrt(yzAlpha)).lw(gridLineWidth * 2)
                highlights.append(hyx)

        if yzGrid and ytitle and ztitle:
            if yHighlightZero and min_bns[2] <= 0 and max_bns[3] > 0:
                yhl = -min_bns[2] / sizes[1]
                hyz = shapes.Line(
                    [0, yhl, 0], [0, yhl, 1], c=yHighlightZeroColor
                )
                hyz.alpha(np.sqrt(yzAlpha)).lw(gridLineWidth * 2)
                highlights.append(hyz)
            if zHighlightZero and min_bns[4] <= 0 and max_bns[5] > 0:
                zhl = -min_bns[4] / sizes[2]
                hzy = shapes.Line(
                    [0, 0, zhl], [0, 1, zhl], c=zHighlightZeroColor
                )
                hzy.alpha(np.sqrt(yzAlpha)).lw(gridLineWidth * 2)
                highlights.append(hzy)

        if zxGrid and ztitle and xtitle:
            if zHighlightZero and min_bns[4] <= 0 and max_bns[5] > 0:
                zhl = -min_bns[4] / sizes[2]
                hzx = shapes.Line(
                    [0, 0, zhl], [1, 0, zhl], c=zHighlightZeroColor
                )
                hzx.alpha(np.sqrt(zxAlpha)).lw(gridLineWidth * 2)
                highlights.append(hzx)
            if xHighlightZero and min_bns[0] <= 0 and max_bns[1] > 0:
                xhl = -min_bns[0] / sizes[0]
                hxz = shapes.Line(
                    [xhl, 0, 0], [xhl, 0, 1], c=xHighlightZeroColor
                )
                hxz.alpha(np.sqrt(zxAlpha)).lw(gridLineWidth * 2)
                highlights.append(hxz)

        x_aspect_ratio_scale = 1
        y_aspect_ratio_scale = 1
        z_aspect_ratio_scale = 1
        if xtitle:
            if sizes[0] > sizes[1]:
                x_aspect_ratio_scale = (1, sizes[0] / sizes[1], 1)
            else:
                x_aspect_ratio_scale = (sizes[1] / sizes[0], 1, 1)

        if ytitle:
            if sizes[0] > sizes[1]:
                y_aspect_ratio_scale = (sizes[0] / sizes[1], 1, 1)
            else:
                y_aspect_ratio_scale = (1, sizes[1] / sizes[0], 1)

        if ztitle:
            smean = (sizes[0] + sizes[1]) / 2
            if smean:
                if sizes[2] > smean:
                    zarfact = smean / sizes[2]
                    z_aspect_ratio_scale = (
                        zarfact,
                        zarfact * sizes[2] / smean,
                        zarfact,
                    )
                else:
                    z_aspect_ratio_scale = (smean / sizes[2], 1, 1)

        titles = []
        if xtitle:
            xt = shapes.Text3D(
                xtitle,
                pos=(0, 0, 0),
                s=xTitleSize,
                c=xTitleColor,
                justify=xTitleJustify,
                depth=titleDepth,
            )
            if reorientShortTitle and len(ytitle) < 3:  # title is short
                wpos = [xTitlePosition, -xTitleOffset + 0.02, 0]
            else:
                wpos = [xTitlePosition, -xTitleOffset, 0]
            if xKeepAspectRatio:
                xt.SetScale(x_aspect_ratio_scale)
            xt.RotateX(xTitleRotation)
            xt.pos(wpos)
            titles.append(xt.lighting(specular=0, diffuse=0, ambient=1))

        if ytitle:
            yt = shapes.Text3D(
                ytitle,
                pos=(0, 0, 0),
                s=yTitleSize,
                c=yTitleColor,
                justify=yTitleJustify,
                depth=titleDepth,
            )
            if reorientShortTitle and len(ytitle) < 3:  # title is short
                wpos = [
                    -yTitleOffset + 0.03 - 0.01 * len(ytitle),
                    yTitlePosition,
                    0,
                ]
                if yKeepAspectRatio:
                    # x!
                    yt.SetScale(x_aspect_ratio_scale)
            else:
                wpos = [-yTitleOffset, yTitlePosition, 0]
                if yKeepAspectRatio:
                    yt.SetScale(y_aspect_ratio_scale)
                yt.RotateZ(yTitleRotation)
            yt.pos(wpos)
            titles.append(yt.lighting(specular=0, diffuse=0, ambient=1))

        if ztitle:
            zt = shapes.Text3D(
                ztitle,
                pos=(0, 0, 0),
                s=zTitleSize,
                c=zTitleColor,
                justify=zTitleJustify,
                depth=titleDepth,
            )
            if reorientShortTitle and len(ztitle) < 3:  # title is short
                wpos = [
                    (-zTitleOffset + 0.02 - 0.003 * len(ztitle)) / 1.42,
                    (-zTitleOffset + 0.02 - 0.003 * len(ztitle)) / 1.42,
                    zTitlePosition,
                ]
                if zKeepAspectRatio:
                    zr2 = (
                        z_aspect_ratio_scale[1],
                        z_aspect_ratio_scale[0],
                        z_aspect_ratio_scale[2],
                    )
                    zt.SetScale(zr2)
                zt.RotateX(90)
                zt.RotateY(45)
                zt.pos(wpos)
            else:
                if zKeepAspectRatio:
                    zt.SetScale(z_aspect_ratio_scale)
                wpos = [
                    -zTitleOffset / 1.42,
                    -zTitleOffset / 1.42,
                    zTitlePosition,
                ]
                zt.RotateY(-90)
                zt.RotateX(zTitleRotation)
                zt.pos(wpos)
            titles.append(zt.lighting(specular=0, diffuse=0, ambient=1))

        originmarks = []
        if originMarkerSize:
            if xtitle:
                if min_bns[0] <= 0 and max_bns[1] > 0:  # mark x origin
                    ox = shapes.Cube(
                        [-min_bns[0] / sizes[0], 0, 0],
                        side=originMarkerSize,
                        c=xLineColor,
                    )
                    originmarks.append(
                        ox.lighting(specular=0, diffuse=0, ambient=1)
                    )

            if ytitle:
                if min_bns[2] <= 0 and max_bns[3] > 0:  # mark y origin
                    oy = shapes.Cube(
                        [0, -min_bns[2] / sizes[1], 0],
                        side=originMarkerSize,
                        c=yLineColor,
                    )
                    originmarks.append(
                        oy.lighting(specular=0, diffuse=0, ambient=1)
                    )

            if ztitle:
                if min_bns[4] <= 0 and max_bns[5] > 0:  # mark z origin
                    oz = shapes.Cube(
                        [0, 0, -min_bns[4] / sizes[2]],
                        side=originMarkerSize,
                        c=zLineColor,
                    )
                    originmarks.append(
                        oz.lighting(specular=0, diffuse=0, ambient=1)
                    )

        cones = []
        if tipSize:
            if xtitle:
                cx = shapes.Cone(
                    (1, 0, 0),
                    r=tipSize,
                    height=tipSize * 2,
                    axis=(1, 0, 0),
                    c=xLineColor,
                    res=10,
                )
                cones.append(cx.lighting(specular=0, diffuse=0, ambient=1))
            if ytitle:
                cy = shapes.Cone(
                    (0, 1, 0),
                    r=tipSize,
                    height=tipSize * 2,
                    axis=(0, 1, 0),
                    c=yLineColor,
                    res=10,
                )
                cones.append(cy.lighting(specular=0, diffuse=0, ambient=1))
            if ztitle:
                cz = shapes.Cone(
                    (0, 0, 1),
                    r=tipSize,
                    height=tipSize * 2,
                    axis=(0, 0, 1),
                    c=zLineColor,
                    res=10,
                )
                cones.append(cz.lighting(specular=0, diffuse=0, ambient=1))

        ticks = []
        if showTicks:
            if xtitle:
                for coo in range(1, rx):
                    v = [coo / rx, 0, 0]
                    xds = shapes.Cylinder(
                        v,
                        r=xTickRadius,
                        height=xTickThickness,
                        axis=(1, 0, 0),
                        res=10,
                    )
                    ticks.append(
                        xds.c(xTickColor).lighting(specular=0, ambient=1)
                    )
            if ytitle:
                for coo in range(1, ry):
                    v = [0, coo / ry, 0]
                    yds = shapes.Cylinder(
                        v,
                        r=yTickRadius,
                        height=yTickThickness,
                        axis=(0, 1, 0),
                        res=10,
                    )
                    ticks.append(
                        yds.c(yTickColor).lighting(specular=0, ambient=1)
                    )
            if ztitle:
                for coo in range(1, rz):
                    v = [0, 0, coo / rz]
                    zds = shapes.Cylinder(
                        v,
                        r=zTickRadius,
                        height=zTickThickness,
                        axis=(0, 0, 1),
                        res=10,
                    )
                    ticks.append(
                        zds.c(zTickColor).lighting(specular=0, ambient=1)
                    )

        minorticks = []
        if xMinorTicks and xtitle:
            xMinorTicks += 1
            for coo in range(1, rx * xMinorTicks):
                v = [coo / rx / xMinorTicks, 0, 0]
                mxds = shapes.Cylinder(
                    v,
                    r=xTickRadius / 1.5,
                    height=xTickThickness,
                    axis=(1, 0, 0),
                    res=6,
                )
                minorticks.append(
                    mxds.c(xTickColor).lighting(specular=0, ambient=1)
                )
        if yMinorTicks and ytitle:
            yMinorTicks += 1
            for coo in range(1, ry * yMinorTicks):
                v = [0, coo / ry / yMinorTicks, 0]
                myds = shapes.Cylinder(
                    v,
                    r=yTickRadius / 1.5,
                    height=yTickThickness,
                    axis=(0, 1, 0),
                    res=6,
                )
                minorticks.append(
                    myds.c(yTickColor).lighting(specular=0, ambient=1)
                )
        if zMinorTicks and ztitle:
            zMinorTicks += 1
            for coo in range(1, rz * zMinorTicks):
                v = [0, 0, coo / rz / zMinorTicks]
                mzds = shapes.Cylinder(
                    v,
                    r=zTickRadius / 1.5,
                    height=zTickThickness,
                    axis=(0, 0, 1),
                    res=6,
                )
                minorticks.append(
                    mzds.c(zTickColor).lighting(specular=0, ambient=1)
                )

        labels = []
        if xLabelSize:
            if xtitle:
                if rx > 12:
                    rx = int(rx / 2)
                for ic in range(1, rx + enableLastLabel):
                    v = (ic / rx, -xLabelOffset, 0)
                    val = v[0] * sizes[0] + min_bns[0]
                    if abs(val) > 1 and sizes[0] < 1:
                        xLabelPrecision = int(
                            xLabelPrecision - np.log10(sizes[0])
                        )
                    tval = precision(val, xLabelPrecision, vrange=sizes[0])
                    xlab = shapes.Text3D(
                        tval,
                        pos=v,
                        s=xLabelSize,
                        justify="center-top",
                        depth=0,
                    )
                    if xKeepAspectRatio:
                        xlab.SetScale(x_aspect_ratio_scale)
                    labels.append(
                        xlab.c(xTickColor).lighting(specular=0, ambient=1)
                    )
        if yLabelSize:
            if ytitle:
                if ry > 12:
                    ry = int(ry / 2)
                for ic in range(1, ry + enableLastLabel):
                    v = (-yLabelOffset, ic / ry, 0)
                    val = v[1] * sizes[1] + min_bns[2]
                    if abs(val) > 1 and sizes[1] < 1:
                        yLabelPrecision = int(
                            yLabelPrecision - np.log10(sizes[1])
                        )
                    tval = precision(val, yLabelPrecision, vrange=sizes[1])
                    ylab = shapes.Text3D(
                        tval,
                        pos=(0, 0, 0),
                        s=yLabelSize,
                        justify="center-bottom",
                        depth=0,
                    )
                    if yKeepAspectRatio:
                        ylab.SetScale(y_aspect_ratio_scale)
                    ylab.RotateZ(yTitleRotation)
                    ylab.pos(v)
                    labels.append(
                        ylab.c(yTickColor).lighting(specular=0, ambient=1)
                    )
        if zLabelSize:
            if ztitle:
                if rz > 12:
                    rz = int(rz / 2)
                for ic in range(1, rz + enableLastLabel):
                    v = (-zLabelOffset, -zLabelOffset, ic / rz)
                    val = v[2] * sizes[2] + min_bns[4]
                    tval = precision(val, zLabelPrecision, vrange=sizes[2])
                    if abs(val) > 1 and sizes[2] < 1:
                        zLabelPrecision = int(
                            zLabelPrecision - np.log10(sizes[2])
                        )
                    zlab = shapes.Text3D(
                        tval,
                        pos=(0, 0, 0),
                        s=zLabelSize,
                        justify="center-bottom",
                        depth=0,
                    )
                    if zKeepAspectRatio:
                        zlab.SetScale(z_aspect_ratio_scale)
                    zlab.RotateY(-90)
                    zlab.RotateX(zTitleRotation)
                    zlab.pos(v)
                    labels.append(
                        zlab.c(zTickColor).lighting(specular=0, ambient=1)
                    )

        acts = grids + grids2 + lines + highlights + titles
        acts += minorticks + originmarks + ticks + cones + labels
        for a in acts:
            a.PickableOff()
        self.grid = Assembly(acts)
        self.grid.pos(min_bns[0], min_bns[2], min_bns[4])
        self.grid.SetScale(sizes)
        self.grid.PickableOff()

    def computeVisibleBounds(self, actors):
        """Calculate max actors bounds and sizes."""
        bns = []
        if actors and actors.GetPickable():
            b = actors.GetBounds()
            if b:
                bns.append(b)
        if len(bns):
            max_bns = np.max(bns, axis=0)
            min_bns = np.min(bns, axis=0)
            vbb = (
                min_bns[0],
                max_bns[1],
                min_bns[2],
                max_bns[3],
                min_bns[4],
                max_bns[5],
            )
        else:
            vbb = settings.plotter_instance.renderer.ComputeVisiblePropBounds()
            max_bns = vbb
            min_bns = vbb
        sizes = np.array(
            [
                max_bns[1] - min_bns[0],
                max_bns[3] - min_bns[2],
                max_bns[5] - min_bns[4],
            ]
        )
        return vbb, sizes, min_bns, max_bns


class Cutter(object):
    """Class using the Cutter attributes."""

    def __init__(self, renderer, interactor, actor):
        """Init function."""
        self.boxWidget = vtk.vtkBoxWidget()
        self.renderer = renderer
        self.interactor = interactor
        self.actor = actor
        self.planes = vtk.vtkPlanes()
        self.properties()
        self.origins = []
        self.normals = []

    def properties(self):
        """Set properties."""
        self.boxWidget.SetInteractor(self.interactor)
        self.boxWidget.SetPlaceFactor(1.0)
        self.boxWidget.SetInputData(self.actor.inputdata())
        self.boxWidget.OutlineCursorWiresOn()
        self.boxWidget.GetSelectedOutlineProperty().SetColor(1, 0, 1)
        self.boxWidget.GetOutlineProperty().SetColor(0.2, 0.2, 0.2)
        self.boxWidget.GetOutlineProperty().SetOpacity(0.7)
        self.boxWidget.SetPlaceFactor(1.0)
        self.boxWidget.PlaceWidget()
        self.boxWidget.InsideOutOn()
        self.boxWidget.AddObserver("InteractionEvent", self.clipVolumeRender)

    def clipVolumeRender(self, obj, event):
        """Set volume render."""
        self.origins = []
        self.normals = []
        obj.GetPlanes(self.planes)
        self.actor.mapper().SetClippingPlanes(self.planes)
        for faces in range(self.planes.GetNumberOfPlanes()):
            self.origins.append(self.planes.GetPlane(faces).GetOrigin())
            self.normals.append(self.planes.GetPlane(faces).GetNormal())


class Mover(object):
    """Class using the Mover attributes."""

    def __init__(self, renderer, interactor, actor):
        """Init function."""
        self.boxWidget = vtk.vtkBoxWidget()
        self.renderer = renderer
        self.interactor = interactor
        self.actor = actor
        self.properties()

    def properties(self):
        """Set properties."""
        # A Box widget
        self.boxWidget = vtk.vtkBoxWidget()
        self.boxWidget.SetInteractor(self.interactor)
        self.boxWidget.SetProp3D(self.actor)
        self.boxWidget.SetPlaceFactor(1.0)
        self.boxWidget.PlaceWidget()

        # Connect the event to a function
        self.boxWidget.AddObserver("InteractionEvent", self.boxCallback)

    def boxCallback(self, obj, event):
        """Get box callback."""
        t = vtk.vtkTransform()
        obj.GetTransform(t)
        obj.GetProp3D().SetUserTransform(t)


def Text_2D(
    txt,
    pos=3,
    s=1,
    c=None,
    alpha=1,
    bg=None,
    font="Montserrat",
    justify="bottom-left",
    bold=False,
    italic=False,
):
    """Return a ``vtkActor2D`` representing 2D text.

    Parameters
    ----------
    pos: text is placed in one of the 8 positions:
        1, bottom-left
        2, bottom-right
        3, top-left
        4, top-right
        5, bottom-middle
        6, middle-right
        7, middle-left
        8, top-middle
        If a pair (x,y) is passed as input the 2D text is place at that
        position in the coordinate system of the 2D screen (with the
        origin sitting at the bottom left).

    pos: list, int

    s: size of text.
    bg: background color
    float alpha: background opacity
    justify: text justification
    font: available fonts are
        - Courier
        - Times
        - Arial
        - CallingCode
        - ChineseRuler
        - Godsway
        - ImpactLabel
        - Komiko
        - Monospace
        - Montserrat
        - Overspray
    """
    if c is None:
        # automatic black or white
        if settings.plotter_instance and settings.plotter_instance.renderer:
            c = (0.9, 0.9, 0.9)
            if settings.plotter_instance.renderer.GetGradientBackground():
                bgcol = settings.plotter_instance.renderer.GetBackground2()
            else:
                bgcol = settings.plotter_instance.renderer.GetBackground()
            if np.sum(bgcol) > 1.5:
                c = (0.1, 0.1, 0.1)
        else:
            c = (0.5, 0.5, 0.5)

    if isinstance(pos, str):
        # corners
        if "top" in pos:
            if "left" in pos:
                pos = 3
            elif "mid" in pos:
                pos = 8
            elif "right" in pos:
                pos = 4
        elif "bottom" in pos:
            if "left" in pos:
                pos = 1
            elif "mid" in pos:
                pos = 5
            elif "right" in pos:
                pos = 2
        else:
            if "left" in pos:
                pos = 7
            elif "right" in pos:
                pos = 6
            else:
                pos = 3

    if isinstance(pos, int):
        # corners
        if pos > 8:
            pos = 8
        if pos < 1:
            pos = 1
        ca = vtk.vtkCornerAnnotation()
        ca.SetNonlinearFontScaleFactor(s / 2.7)
        ca.SetText(pos - 1, str(txt))
        ca.PickableOff()
        cap = ca.GetTextProperty()
        cap.SetColor(getColor(c))
        if font.lower() == "courier":
            cap.SetFontFamilyToCourier()
        elif font.lower() == "times":
            cap.SetFontFamilyToTimes()
        elif font.lower() == "arial":
            cap.SetFontFamilyToArial()
        else:
            cap.SetFontFamily(vtk.VTK_FONT_FILE)
            cap.SetFontFile(settings.fonts_path + font + ".ttf")
        if bg:
            bgcol = getColor(bg)
            cap.SetBackgroundColor(bgcol)
            cap.SetBackgroundOpacity(alpha * 0.1)
            # cap.SetFrameColor(bgcol)
            # cap.FrameOn()
        cap.SetBold(bold)
        cap.SetItalic(italic)
        setattr(ca, "renderedAt", set())
        settings.collectable_actors.append(ca)

        ###############
        return ca
        ###############

    if len(pos) != 2:
        print(
            "Error in Text2D():"
            + "len(pos) must be 2 or integer value or string."
        )
        raise RuntimeError()

    else:
        actor2d = vtk.vtkActor2D()
        actor2d.GetPositionCoordinate()
        actor2d.SetCoordinateSystemToNormalizedViewport()
        actor2d.SetPosition(pos)
        tmapper = vtk.vtkTextMapper()
        tmapper.SetInput(str(txt))
        actor2d.SetMapper(tmapper)
        tp = tmapper.GetTextProperty()
        tp.BoldOff()
        tp.SetFontSize(int(s * 20))
        tp.SetColor(getColor(c))
        tp.SetJustificationToLeft()
        if "top" in justify:
            tp.SetVerticalJustificationToTop()
        if "bottom" in justify:
            tp.SetVerticalJustificationToBottom()
        if "cent" in justify:
            tp.SetVerticalJustificationToCentered()
            tp.SetJustificationToCentered()
        if "left" in justify:
            tp.SetJustificationToLeft()
        if "right" in justify:
            tp.SetJustificationToRight()

        if font.lower() == "courier":
            tp.SetFontFamilyToCourier()
        elif font.lower() == "times":
            tp.SetFontFamilyToTimes()
        elif font.lower() == "arial":
            tp.SetFontFamilyToArial()
        else:
            tp.SetFontFamily(vtk.VTK_FONT_FILE)
            if font in settings.fonts:
                tp.SetFontFile(settings.fonts_path + font + ".ttf")
            elif os.path.exists(font):
                tp.SetFontFile(font)
            else:
                tp.SetFontFamilyToCourier()
                # silently fail
        if bg:
            bgcol = getColor(bg)
            tp.SetBackgroundColor(bgcol)
            tp.SetBackgroundOpacity(alpha * 0.1)
            tp.SetFrameColor(bgcol)
            tp.FrameOn()
        actor2d.PickableOff()
        setattr(actor2d, "renderedAt", set())
        settings.collectable_actors.append(actor2d)
        return actor2d
