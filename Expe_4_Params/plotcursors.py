"""
===========
Cursor Demo
===========

This example shows how to use Matplotlib to provide a data cursor.  It uses
Matplotlib to draw the cursor and may be a slow since this requires redrawing
the figure with every mouse move.

Faster cursoring is possible using native GUI drawing, as in
:doc:`/gallery/user_interfaces/wxcursor_demo_sgskip`.

Numbers display improved by G. Le Vaillant, 2019
"""

import matplotlib.pyplot as plt
import numpy as np
from decimal import Decimal


class CursorBase:
    """
    Base pour les curseurs constraints et non-constraints sur la courbe
    """
    def __init__(self, ax, x_data, y_data, text_x, text_y, use_scientific_notation):
        self.ax = ax

        # x,y data is required for the cursor to track the curve... and for init
        self.x_data = x_data
        self.y_data = y_data

        # and they should be sorted... SORT TO IMPLEMENT HERE
        self.current_x, self.current_y = x_data[0], y_data[0]

        # text location in axes coords
        self.txt = ax.text(text_x, text_y, '', transform=ax.transAxes)
        # text display params
        self.use_scientific_notation = use_scientific_notation

        self.lx = ax.axhline(y=self.current_y, color='k', linewidth=0.5, linestyle="--")  # the horiz line
        self.ly = ax.axvline(color='k', linewidth=0.5, linestyle="--")  # the vert line

        self.update_display()

    """ Updates the display. x,y data must be updated before calling this function. """
    def update_display(self):
        # update the line positions
        self.lx.set_ydata(self.current_y)
        self.ly.set_xdata(self.current_x)
        # scientific notation ?
        if self.use_scientific_notation:
            self.txt.set_text('x={:.4E}, y={:.4E}'.format(Decimal(self.current_x), Decimal(self.current_y)))
        else:
            self.txt.set_text('x={:1.3f}, y={:1.3f}'.format(self.current_x, self.current_y))

        self.ax.figure.canvas.draw()


class Cursor(CursorBase):
    def __init__(self, ax, x_data, y_data, text_x=0.7, text_y=0.9, use_scientific_notation=False):
        CursorBase.__init__(self, ax, x_data, y_data, text_x, text_y, use_scientific_notation)

    def mouse_move(self, event):
        if not event.inaxes:
            return
        self.current_x, self.current_y = event.xdata, event.ydata
        super().update_display()


class SnapCursor(CursorBase):
    """
    Like Cursor but the crosshair snaps to the nearest x, y point.
    For simplicity, this assumes that *x* is sorted.
    """

    def __init__(self, ax, x_data, y_data, text_x=0.7, text_y=0.9, use_scientific_notation=False):
        CursorBase.__init__(self, ax, x_data, y_data, text_x, text_y, use_scientific_notation)

    def mouse_move(self, event):
        if not event.inaxes:
            return
        x, y = event.xdata, event.ydata
        index = min(np.searchsorted(self.x_data, x), len(self.x_data) - 1)
        self.current_x = self.x_data[index]
        self.current_y = self.y_data[index]
        super().update_display()


if __name__ == '__main__':
    t = np.arange(0.0, 1.0, 0.01)
    s = np.sin(2 * 2 * np.pi * t) * 1000.0

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(t, s, 'o')
    cursor = Cursor(ax1, t, s)
    fig.canvas.mpl_connect('motion_notify_event', cursor.mouse_move)

    ax2.plot(t, s, 'o')
    snap_cursor = SnapCursor(ax2, t, s, 0.1, 0.1, use_scientific_notation=True)
    fig.canvas.mpl_connect('motion_notify_event', snap_cursor.mouse_move)

    plt.show()
