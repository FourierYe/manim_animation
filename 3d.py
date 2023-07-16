import numpy as np
from manim import *

class ThreeDLightSourcePosition(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes()
        sphere = Surface(
            lambda u, v: np.array([
                u,v,(np.sin(u)+np.cos(v))**2+u -u/v
            ]), v_range=[0, TAU], u_range=[-PI / 2*3, PI / 2*3], resolution=(15, 32)
        )
        self.renderer.camera.light_source.move_to(3*IN) # changes the source of the light
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        sphere.set_fill_by_value(axes=axes, colorscale=[], axis=2)
        self.play(Create(axes), Create(sphere))
        self.interactive_embed()