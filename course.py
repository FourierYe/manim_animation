import numpy as np
from manim import *

class Main(Scene):
    def construct(self):
        # self.show_info()
        # self.show_title()
        # self.plot_linear_regression()
        # self.gd_overview()
        # self.gd_process()
        # self.gd_example()
        # self.gd_play()
        # self.explain_gd()
        self.gd_lr()

        # self.thanks()
    def show_info(self):
        name = Text("叶志鹏")
        department = Text("计算机科学与工程学院")
        school = Text("南京理工大学泰州科技学院")
        info = VGroup(school, department, name).arrange(DOWN).center()
        self.play(Create(info), run_time=3)
        self.wait(3)
        self.play(FadeOut(info))

    def show_title(self):
        title = Text("第十章-线性回归")
        subtitle = Text("梯度下降法")
        info = VGroup(title, subtitle).arrange(DOWN).center()
        self.play(Write(info), run_time=3)
        self.wait(3)
        self.play(FadeOut(info))

    def thanks(self):
        thanks = Text("Thanks")
        self.play(Write(thanks), run_time=3)
        self.wait(3)

    def plot_linear_regression(self):
        review = Text("上集回顾").to_edge(UL)
        self.add(review)

        argmse = MathTex("argmin_{W,b} \sum_{i=1}^{N}(WX_{i}+b-Y_{i})", font_size=38)

        solver = VGroup(Tex(r"解析法求均方误差函数的最小值是使$MSE(W,b)$", font_size=26, tex_template=TexTemplateLibrary.ctex),
                        Tex(r"各方向上的偏导数等于$0$联立方程组,", font_size=26, tex_template=TexTemplateLibrary.ctex),
                        Tex(r"并根据二阶导数性质判断极值点来求极小值点。", font_size=26, tex_template=TexTemplateLibrary.ctex)
                        ).arrange(DOWN, aligned_edge=LEFT)

        sentence = VGroup(Tex(r"当$W$是多维向量时，通过解析法", font_size=26, tex_template=TexTemplateLibrary.ctex),
                          Tex(r"求$MSE(W, b)$函数的最小值是非常难计算的！", font_size=26, tex_template=TexTemplateLibrary.ctex)
                          ).arrange(DOWN, aligned_edge=LEFT)

        info = VGroup(argmse, solver, sentence).arrange(DOWN, aligned_edge=LEFT).to_edge(RIGHT)

        def fun_(x):
            return 2*x+3

        ax = Axes(x_range=[-1,3,1], y_range=[-1,9,2], x_length=6, y_length=4).add_coordinates()

        X = np.random.uniform(-1,3,10)
        noise = np.random.normal(0,1,10)
        Y = fun_(X)+noise

        dots = [Dot(ax.coords_to_point(float(x),float(y))) for x,y in zip(X,Y)]
        graph = ax.plot(function=fun_, x_range=[-1,3,1])
        figure = VGroup(ax, *dots, graph).to_edge(LEFT)

        self.play(Create(figure), Write(info))
        self.wait(2)

        a = ValueTracker(0)
        b = ValueTracker(0)
        graph.add_updater(
            lambda m:m.become(
                ax.plot(
                lambda x: a.get_value()*x + b.get_value(),
                    x_range=[-1, 3, 1])
            )
        )
        self.add(graph)
        self.play(ApplyMethod(a.increment_value, 4), ApplyMethod(b.increment_value, 6), run_time=10)
        self.play(ApplyMethod(a.increment_value, -2), ApplyMethod(b.increment_value, -3), run_time=10)

        self.play(FadeOut(figure, info, review))

    def gd_overview(self):
        gd_ = Text("梯度下降法").to_edge(UL)

        gdgif = ImageMobject('gd.png', scale_to_resolution=2160).to_edge(LEFT)
        gdgif.height = 3.5
        self.add(gd_, gdgif)

        info1 = Text("梯度下降法是一种数值优化算法，\n通过迭代的方式求解函数的局部\n极值问题。", font_size=30, line_spacing=0.5)
        info2 = Text("数学或工程上涉及多元非凸函数\n时，常采用梯度下降法求解函数\n的局部极小值，如本章学校的线\n性回归问题"
                     "以及后续的神经网\n络知识点。", font_size=30, line_spacing=0.5).align_to(info1, LEFT)

        info = VGroup(info1, info2).arrange(DOWN).to_edge(RIGHT)
        self.play(Write(info), run_time=10)
        self.wait(3)

        self.play(FadeOut(gd_, gdgif, info1, info2))

    def gd_process(self):

        gd_ = Text("梯度下降法步骤").to_edge(UL)

        step1_1 = Tex(r"1. 随机初始化函数$f_{W,b}(X)$的参数$W,b$。设定算法的终止条件", font_size=30,
                    tex_template=TexTemplateLibrary.ctex)
        step1_2 = Tex(r"（梯度变化范围或迭代次数），设定算法的学习率 learning rate $\alpha$", font_size=30,
                    tex_template=TexTemplateLibrary.ctex)

        step1 = VGroup(step1_1, step1_2).arrange(DOWN, aligned_edge=LEFT)

        step2 = Tex(r"2. 计算当前函数的梯度，"
                      r"$\nabla f = <\frac{\partial f}{\partial W}, \frac{\partial f}{\partial b}>|_{(W,b)}$", font_size=30,
                    tex_template=TexTemplateLibrary.ctex)
        step3_1 = Tex(r"3. 更新参数 $W, b$，即", font_size=30,
                    tex_template=TexTemplateLibrary.ctex)
        step3_2 = MathTex(r"""& W \leftarrow W - \alpha*\frac{\partial f}{\partial W}|_{(W,b)}\\
                & b \leftarrow b - \alpha*\frac{\partial f}{\partial b}|_{(W,b)}""", font_size=30,
                    tex_template=TexTemplateLibrary.ctex)

        step3 = VGroup(step3_1, step3_2).arrange(DOWN, aligned_edge=LEFT)

        step4 = Tex(r"4. 判断终止条件，如不符合条件，跳到步骤二；符合条件算法终止。", font_size=30,
                    tex_template=TexTemplateLibrary.ctex)

        steps = VGroup(step1, step2, step3, step4).arrange(DOWN, aligned_edge=LEFT).to_edge(LEFT)

        all = VGroup(gd_, steps).arrange(DOWN, aligned_edge=LEFT).to_corner(UL)

        self.play(Write(all), run_time=10)

        self.wait(30)
        self.play(FadeOut(all))

    def gd_example(self):
        question = Tex(r"请使用梯度下降法，求函数$f(x)=x^2+4x+8$的最小值。", font_size=30,
                    tex_template=TexTemplateLibrary.ctex)
        self.play(Write(question), run_time=2)
        self.play(question.animate.to_corner(UL), run_time=2)
        self.wait(3)

        config = Tex(r"设学习率learning rate $\alpha = 0.2$, 迭代$10$次", font_size=30,
                    tex_template=TexTemplateLibrary.ctex).next_to(question, DOWN).to_edge(LEFT)
        self.play(Write(config), run_time=2)

        steps = np.arange(0, 11, 1)
        x = 2
        def fx(x):
            return x**2+4*x+8
        def fxx(x):
            return 2*x+4

        lr = 0.2
        X = [x]
        FX = [fx(x)]
        FXX = [fxx(x)]
        for step in steps:
            x = x - lr * fxx(x)
            X.append(x)
            FX.append(fx(x))
            FXX.append(fxx(x))

        table = DecimalTable([steps, X[:-1], FX[:-1], FXX[:-1], X[1:]],
                          row_labels=[MathTex("step"), MathTex("x"), MathTex('f(x)'), MathTex("f'(x)"), MathTex("x - lr*f'(x)")],
                          h_buff=1,
                          element_to_mobject_config={"num_decimal_places": 5})
        table.width = 14
        self.play(Create(table.get_labels()))
        self.play(Create(table.get_horizontal_lines()))

        self.wait(10)
        col1 = table.get_columns()[1]
        for i in range(1, 6):
            cell = table.get_cell((i, 2))
            self.play(Create(cell), Create(col1[i-1]))
            self.wait(3)
            self.play(FadeOut(cell))
        self.wait(5)
        col2 = table.get_columns()[2]
        for i in range(1, 6):
            cell = table.get_cell((i, 3))
            self.play(Create(cell), Create(col2[i-1]))
            self.wait(3)
            self.play(FadeOut(cell))
        self.play(
            AnimationGroup(
                *[Create(i) for i in table.get_columns()[3:]],
                lag_ratio=0.8
            ),
            run_time=10
        )
        self.wait(10)
        self.clear()
        self.wait(2)

    def gd_play(self):
        gd_ = Text("梯度下降法可视化").to_corner(UL)
        self.play(Write(gd_))

        ax = Axes(x_range=[-7, 3, 1], y_range=[-1, 31, 5], x_length=6, y_length=4).add_coordinates()
        graph = ax.plot(lambda x: x**2+4*x+8, x_range=[-7, 3, 1])
        fun_ = MathTex(r"f(x)=x^2+4x+8")
        fun_.next_to(ax, UP)
        self.play(Write(fun_), Create(ax), Create(graph))
        self.wait(2)
        self.play(FadeOut(gd_))
        figure = VGroup(fun_, ax, graph)
        self.play(figure.animate.to_edge(LEFT))
        self.play(figure.animate.scale(1.2))

        steps = np.arange(0, 11, 1)
        x = 2

        def fx(x):
            return x ** 2 + 4 * x + 8

        def fxx(x):
            return 2 * x + 4

        lr = 0.2
        X = [x]
        FX = [fx(x)]
        FXX = [fxx(x)]
        for step in steps:
            x = x - lr * fxx(x)
            X.append(x)
            FX.append(fx(x))
            FXX.append(fxx(x))

        step = ValueTracker(0)

        dot = Dot(ax.coords_to_point(X[0], FX[0]), color=RED)
        line = ax.plot(lambda x: (x-X[0])*FXX[0]+fx(X[0]) , x_range=[-7, 3, 1])

        dot.add_updater(
            lambda m:m.become(
                Dot(ax.coords_to_point(X[int(step.get_value())], FX[int(step.get_value())]), color=RED)
            )
        )
        line.add_updater(
            lambda m:m.become(
                ax.plot(
                lambda x: FXX[int(step.get_value())]*(x-X[int(step.get_value())]) + FX[int(step.get_value())], x_range=[-7, 3, 1])
            )
        )
        self.play(Create(line), Create(dot))

        learning_rate = Tex(f"学习率 lr = {lr:.2f}", tex_template=TexTemplateLibrary.ctex)
        step_info = Tex(f"迭代次数：{0}",tex_template=TexTemplateLibrary.ctex)
        x_info = Tex(f"当前$x$：{X[0]:.2f}",tex_template=TexTemplateLibrary.ctex)
        fx_info = Tex(f"当前$f(x)$: {FX[0]:.2f}",tex_template=TexTemplateLibrary.ctex)
        fxx_info = Tex(f"当前$f'(x)$: {FXX[0]:.2f}",tex_template=TexTemplateLibrary.ctex)
        formula = MathTex(f"x-lr*f'(x): {X[1]:.2f}", tex_template=TexTemplateLibrary.ctex)
        info = always_redraw(lambda : VGroup(learning_rate, step_info, x_info, fx_info, fxx_info, formula).arrange(DOWN, aligned_edge=LEFT).to_edge(RIGHT))
        self.play(Write(info))

        def info_update_text(mob):

            learning_rate = Tex(f"学习率 lr = {lr}", tex_template=TexTemplateLibrary.ctex)
            step_info = Tex(f"迭代次数：{int(steps[int(step.get_value())])}",tex_template=TexTemplateLibrary.ctex)
            x_info = Tex(f"当前$x$：{X[int(step.get_value())]:.2f}",tex_template=TexTemplateLibrary.ctex)
            fx_info = Tex(f"当前$f(x)$: {FX[int(step.get_value())]:.2f}",tex_template=TexTemplateLibrary.ctex)
            fxx_info = Tex(f"当前$f'(x)$: {FXX[int(step.get_value())]:.2f}",tex_template=TexTemplateLibrary.ctex)
            formula = MathTex(f"x-lr*f'(x):{X[int(step.get_value())]-lr*FXX[int(step.get_value())]:.2f}", tex_template=TexTemplateLibrary.ctex)

            mob.become(
                VGroup(learning_rate, step_info, x_info, fx_info, fxx_info, formula)
                .arrange(DOWN, aligned_edge=LEFT).to_edge(RIGHT))

        info.add_updater(info_update_text)

        self.play(ApplyMethod(step.increment_value, 10), run_time=30)
        self.wait(5)
        self.clear()
        self.wait(2)

    def explain_gd(self):
        title = Text("梯度下降法的直观理解").to_corner(UL)
        self.play(Write(title))

        plane = NumberPlane(x_range=[-4, 6, 1], y_range=[-5, 20, 5], x_length=8, y_length=5).add_coordinates()
        graph = plane.plot(lambda x: (x+3)*(x-1)*(x-3), x_range=[-4, 6, 1])
        self.play(Write(plane), Write(graph))
        self.wait(2)


        def fxx(x):
            return 3*x**2 - 2*x - 9

        def fx(x):
            return x**3 - x**2 - 9*x + 9

        # 右边
        x1 = 3.5
        line = plane.plot(lambda x: fxx(x1)*(x-x1) + fx(x1), x_range=[-4, 6, 1])
        dot = Dot(plane.coords_to_point(x1, fx(x1)), color=RED)

        self.play(Create(line), Create(dot))
        self.wait(5)

        info = Tex(f"x={x1},f'(x)={fxx(x1)}, +", tex_template=TexTemplateLibrary.ctex, font_size=20).next_to(dot, RIGHT)
        self.play(Write(info))
        self.wait(10)
        direction = Line(dot.get_center(), dot.get_center()+LEFT).add_tip()
        x_dir = Tex("$x-lr_{+}*val_{+}$", font_size=30, color=RED).next_to(direction, UP)
        self.play(FadeIn(direction),FadeIn(x_dir), run_time=2)
        self.wait(5)

        self.play(FadeOut(line), FadeOut(info), FadeOut(dot), FadeOut(direction), FadeOut(x_dir))
        self.wait(2)

        # 左边
        x2 = -0.5
        line = plane.plot(lambda x: fxx(x2) * (x - x2) + fx(x2), x_range=[-4, 6, 1])
        dot = Dot(plane.coords_to_point(x2, fx(x2)), color=RED)

        self.play(Create(line), Create(dot))
        self.wait(5)

        info = Tex(f"x={x2},f'(x)={fxx(x2)}, -", tex_template=TexTemplateLibrary.ctex, font_size=20).next_to(dot, LEFT)
        self.play(Write(info))
        self.wait(10)
        direction = Line(dot.get_center(), dot.get_center() + RIGHT).add_tip()
        x_dir = Tex("$x-lr_{+}*val_{-}$", font_size=30, color=RED).next_to(direction, UP)
        self.play(FadeIn(direction), FadeIn(x_dir), run_time=2)
        self.wait(5)

        self.play(FadeOut(line), FadeOut(info), FadeOut(dot), FadeOut(direction), FadeOut(x_dir))
        self.wait(2)

        # 步长
        x3 = 1.2
        line = plane.plot(lambda x: fxx(x3) * (x - x3) + fx(x3), x_range=[-4, 6, 1])
        dot = Dot(plane.coords_to_point(x3, fx(x3)), color=RED)

        self.play(Create(line), Create(dot))
        self.wait(5)

        info = Tex(f"x={x3},f'(x)={fxx(x3)}, -", tex_template=TexTemplateLibrary.ctex, font_size=20).next_to(dot, LEFT)
        self.play(Write(info))
        self.wait(10)
        direction = Line(dot.get_center(), dot.get_center() + 0.5*RIGHT).add_tip()
        x_dir = Tex("$x-lr_{+}*val_{-}$", font_size=30, color=RED).next_to(direction, UP)
        self.play(FadeIn(direction), FadeIn(x_dir), run_time=2)
        self.wait(5)

        self.play(FadeOut(line), FadeOut(info), FadeOut(dot), FadeOut(direction), FadeOut(x_dir))
        self.wait(2)

        self.clear()

    def gd_lr(self):
        title = Text("梯度下降法的直观理解").to_corner(UL)
        self.play(Write(title))