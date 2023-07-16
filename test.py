from manim import *
class FadingTableCellHighlight(Scene):
    def construct(self):
        t = Table(
              [["Red", "."   , "."   ],
              ["."  ,"Green", "X"   ],
              ["."  ,"."    , "Blue"]],
              row_labels=[Text("R1"), Text("R2"), Text("R3")],
              col_labels=[Text("C1"), Text("C2"), Text("C3")],
              top_left_entry=Text("TOP")
            )

        t.add_highlighted_cell((2, 2), color=PURE_RED)    # opacity of 1
        t.add_highlighted_cell((3, 3), color=PURE_GREEN)  # opacity of 0.25
        t.add_highlighted_cell((4, 4), color=PURE_BLUE)   # opacity of 0.5
        t.add_highlighted_cell((3, 4), color=YELLOW)      # opacity of 0.2
        # the t[i] represents the ith add_highlighted_cell call in reverse
        # order. So, the last highlighted cell will have index 0, the second
        t[0].set_opacity(0.2)   # yellow
        t[1].set_opacity(0.25)  # blue
        t[2].set_opacity(0.5)   # green
        t[3].set_opacity(0)     # red

        self.add(t)
        # fades out the selection with a duration of five seconds
        self.play(t[3].animate.set_opacity(1),run_time=5)
        self.wait()