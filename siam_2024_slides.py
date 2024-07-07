from typing import *

import numpy as np

from manim import * 
from manim_slides import Slide


class ScatteringScene:
    """A class representing a scattering scene"""
    def __init__(
        self, 
        centers: List[Tuple[float, float]], 
        obstacle_boundaries: List[List[Callable]], 
        artificial_radii: List[float],
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float
    ):
        """Set up the scattering scene. CAN HANDLE AT MOST 4 OBSTACLES
        
        Parameters:
            centers: A list of center coordinates of obstacles 
            obstacle_boundaries: A list of parametric descriptions
                of 2-dimensional obstacles (DEFINED FROM 0 TO 2pi/TAU),
                e.g.
                     [ [lambda t: np.cos(t), lambda t: np.sin(t)], ...]
            artificial_radii: A list of the radii of the artificial boundaries
                of each obstacle 
            x_min: The minimum x-coordinate on these axes 
            x_max: The maximum x-coordinate on these axes
            y_min: The minimum y-coordinate on these axes 
            y_max: The maximum y-coordinate on these axes
        """
        self.COLORS = [PURPLE, RED, ORANGE, BLUE]
        self.alpha_inc = ValueTracker(1)   # For when we animate incident waves
        self.alpha_sc = ValueTracker(0)    # For when we animate scattered waves
        self.scattered_wave_color = PURE_BLUE 
        self.incident_wave_color = DARK_BROWN

        self.centers = centers
        self.obstacle_boundaries = obstacle_boundaries
        self.artificial_radii = artificial_radii
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

        self.scene = self._set_up_scene()
    
    ###### --------------- SET UP SCENE OBJECTS ------------------ ######
    def _set_up_axis(self):
        """Set up the axis object for plotting"""
        x_step = 1
        y_step = 1
        ax = Axes(
            x_range=[self.x_min, self.x_max, x_step],
            y_range=[self.y_min, self.y_max, y_step],
            x_length=7,
            y_length=7,
            tips=True
        ).set_color(BLACK)
        return ax

    
    def _add_obstacle_curves(self, ax):
        """Add physical obstacles as filled parametric curves on an axis"""
        scatterers = VGroup()
        for center, boundary, color in zip(self.centers, self.obstacle_boundaries, self.COLORS):
            f = lambda t: (center[0] + boundary[0](t), center[1] + boundary[1](t), 0)
            eq = ax.plot_parametric_curve(
                f,
                t_range=[0, TAU],
                color=color,
                fill_opacity=1
            )
            scatterers.add(eq)
        return scatterers

    
    def _add_artificial_boundaries(self, ax):
        """Add artificial boundaries around the scatterers"""
        art_boundaries = VGroup()
        for center, r, color in zip(self.centers, self.artificial_radii, self.COLORS):
            art_bndry_curve = lambda t: (center[0] + r*np.cos(t), center[1] + r*np.sin(t), 0)
            art_bndry_region = ax.plot_parametric_curve(
                art_bndry_curve,
                t_range=[0, TAU],
                color=color,
                fill_opacity=0.2,
                stroke_width=0
            ).set_z_index(-2)
            # art_bndry_dashed = DashedVMobject(art_bndry_orig, fill_opacity=0.3).set_z_index(-2)
            art_bndry_dashed = DashedVMobject(
                ax.plot_parametric_curve(
                    art_bndry_curve,
                    t_range=[0, TAU],
                    color=color,
                )
            ).set_z_index(-2)
            art_bndry = VGroup(*[art_bndry_region, art_bndry_dashed])
            art_boundaries.add(art_bndry)
        return art_boundaries

    
    def _add_labels(self, ax, obstacles, artificial_boundaries):
        """Add the labels Gamma_m (at center of obstacle) and C_m (at artificial boundary) 
        for each of these obstacles."""
        obstacle_labels = VGroup()
        artificial_bndry_labels = VGroup()
        domain_labels = VGroup()
        for i, (obstacle, bndry) in enumerate(zip(obstacles, artificial_boundaries)):
            m = i+1
            gamma_label = (
                MathTex(
                    rf'\Gamma_{m}',
                    font_size=30,
                    color=BLACK
                ).move_to(obstacle)
                .set_z_index(1)
            )
            obstacle_labels.add(gamma_label)

            c_label = (
                MathTex(
                    rf'\mathcal{{C}}_{m}',
                    font_size=30,
                    color=BLACK
                ).next_to(bndry, UP)
            )
            artificial_bndry_labels.add(c_label)

            omega_label = (
                MathTex(
                    rf'\Omega_{m}^-',
                    font_size=30,
                    color=BLACK
                ).next_to(obstacle,RIGHT)
            )
            domain_labels.add(omega_label)

        # Add labels for problem (unbounded domain Omega, etc.)
        omega_label = (
            MathTex(
                r'\Omega^+',
                font_size=30,
                color=BLACK
            ).next_to(ax, RIGHT + UP)
            .shift(DOWN)
        )
        domain_labels.add(omega_label)

        orig_omega_label = (
            MathTex(
                r'\Omega',
                font_size=30,
                color=BLACK
            ).next_to(ax, RIGHT + UP)
            .shift(DOWN)
        )
        

        return obstacle_labels, artificial_bndry_labels, domain_labels, orig_omega_label

    
    def _add_incident_wave(self, ax):
        """Add an incident wave which comes in from the top.""" 
        
        def wave_func_line(b):
            """Produce a plane wave line on the axis with y-intercept b"""
            def wrapper():
                return ax.plot(
                    lambda x: x + self.alpha_inc.get_value() + b,
                    color=self.incident_wave_color,
                    x_range=(
                        max(self.x_min, self.y_min - b - self.alpha_inc.get_value()),
                        min(self.x_max, self.y_max - b - self.alpha_inc.get_value())
                    ),
                    stroke_opacity=0.6
                )
            return wrapper

        plane_wave_funcs = [
            always_redraw(wave_func_line(b))
            for b in range(self.y_min-10, self.y_max+10)
        ]
        
        for func in plane_wave_funcs:
            func.set_z_index(-2)    # Send behind obstacles and scattered waves
        
        return VGroup(*plane_wave_funcs)

        
    def _add_scattered_wave(self, ax):
        """Add scattered waves from each obstacle"""
        
        def scattered_wave_circle(r, center, art_bndry_radius):
            """Produce a scattered wave circle centered at the obstacle with radius r"""
            def wrapper():
                return Circle(
                    radius=r + self.alpha_sc.get_value(),
                    color=self.scattered_wave_color,
                    fill_opacity=0.0,
                    stroke_width=4,
                    stroke_opacity=0.96 * (art_bndry_radius -(r + self.alpha_sc.get_value()))
                ).move_to(ax.coords_to_point(*center))
            return wrapper

        scattered_wave_funcs = VGroup()     # For arranging all at the same time
        for center, R in zip(self.centers, self.artificial_radii):
            wave_funcs = [
                always_redraw(scattered_wave_circle(r/4, center, R))
                for r in range(1,13)
            ]
            for func in wave_funcs:
                func.set_z_index(-1)     # Send behind obstacles, but in front of scattered wave
            wave_func_group = VGroup(*wave_funcs)
            scattered_wave_funcs += wave_func_group

        return scattered_wave_funcs


    def _add_wave_labels(self, ax):
        """Add labels for incident/scattered waves to the LEFT of the diagram"""
        u_sc_label = (
            MathTex(
                r'u',
                color=self.scattered_wave_color
            )
        )

        u_inc_label = (
            MathTex(
                r'u_{inc}',
                color=self.incident_wave_color
            )
        )
        wave_labels = (
            VGroup(u_sc_label, u_inc_label)
            .arrange(direction=UP)   # Arrange them vertically and centered 
            .scale(1.1)
            .next_to(ax, LEFT*2)     # Put to the left of the axis
        )
        return wave_labels 

      
    def _set_up_scene(self):
        """Set up acoustic scattering scene"""
        ax = self._set_up_axis()
        obstacles = self._add_obstacle_curves(ax)
        artificial_boundaries = self._add_artificial_boundaries(ax)
        labels = self._add_labels(ax, obstacles, artificial_boundaries)
        obstacle_labels, artificial_bndry_labels, domain_labels, orig_omega_label = labels
        incident_wave = self._add_incident_wave(ax)
        scattered_wave = self._add_scattered_wave(ax)
        wave_labels = self._add_wave_labels(ax)

        
        return VDict(
            [('ax', ax), 
             ('obstacles', obstacles),
             ('artificial_boundaries', artificial_boundaries),
             ('obstacle_labels', obstacle_labels),
             ('artificial_boundary_labels', artificial_bndry_labels),
             ('domain_labels', domain_labels),
             ('orig_omega_label', orig_omega_label),
             ('incident_wave', incident_wave),
             ('scattered_wave', scattered_wave),
             ('wave_labels', wave_labels)]
        )


    ###### --------------- ANIMATION METHODS ------------------ ######
    def get_incident_wave_animations(self):
        """Get the animation objects which will animate the waves"""
        incident_wave_animation = self.alpha_inc.animate.set_value(0)
        scattered_wave_animation = self.alpha_sc.animate.set_value(0.25)
        return incident_wave_animation, scattered_wave_animation 


    def reset_animation(self):
        """Reset all animation value trackers to their original values"""
        self.alpha_inc.set_value(1)
        self.alpha_sc.set_value(0)
        


class PresentationBase(MovingCameraScene):
    def add_to_tex_template(self):
        myBaseTemplate = TexTemplate(
            documentclass="\documentclass[preview]{standalone}"
        )
        myBaseTemplate.add_to_preamble(r"\usepackage{ragged2e}")
        self.custom_tex_template = myBaseTemplate

    
    def color_latex(self, eqn, tex_to_color_map):
        """Color specific symbols at specific indexes"""
        for substr, info in tex_to_color_map.items():
            for index_pair in info['indexes']:
                outer_index = index_pair[0]
                for inner_index in index_pair[1]:
                    eqn[outer_index][inner_index].set_color(info['color'])
        return eqn


    def get_scattering_scene(self):
        scatterer_boundaries = [
            [lambda t: (1 + 0.2*np.cos(3*t + np.pi)) * np.cos(t), lambda t: (1 + 0.2*np.cos(3*t + np.pi))*np.sin(t)],
            [lambda t: (1 + 0.2*np.cos(7*t + np.pi)) * np.cos(t), lambda t: (1 + 0.2*np.cos(7*t + np.pi))*np.sin(t)],
            [lambda t: (1 + 0.2*np.cos(4*t + np.pi)) * np.cos(t-(np.pi/4)), lambda t: (1 + 0.2*np.cos(4*t + np.pi))*np.sin(t-np.pi/4)]
        ]
        centers = [(2.5, 3.5), (3,-3.5), (-2.2,-1.5)]
        artificial_radii = [2.2, 2.2, 2.2]
        xmin = -5.5
        xmax = 6.5
        ymin = -7
        ymax = 7

        return ScatteringScene(centers, scatterer_boundaries, artificial_radii, xmin, xmax, ymin, ymax)

    
    def roadmap_slide(self):
        # -------------------------- ROADMAP SLIDE --------------------------
        # Add text to slide 
        outline_title = (
            Text(
                "Outline",
                font_size=60,
                color=BLACK
            ).to_edge(UP)
        )
        bullets = (Tex(
                '\item Overview of Problem',
                '\item Formulation of Boundary Conditions',
                '\item Formulation of Iterative Method',
                '\item Results',
                '\item Future Work',
                tex_environment='enumerate'
            ).set_color(BLACK)
        )

        # Return text
        return VDict([('roadmap_title', outline_title), ('roadmap_sections', bullets)])



class Presentation(Slide, PresentationBase):
    ## ........... INTRODUCTION AND ROADMAP SLIDES ........... ##
    def title_slide(self):
        # -------------------------- TITLE SLIDE --------------------------
        # 1) Get text
        title = Paragraph(r"An Iterative Approach to Multiple-Obstacle",
                          "Acoustic Scattering Problems",
                          line_spacing=1.25,
                          alignment='center')
        date = Tex("July 7, 2024")
        author = Tex(r"\textbf{Jordan Sheppard}, MS Student, Brigham Young University")
        university = Tex("Vianey Villamizar, Professor, Brigham Young University")

        # 2) Set format
        # Title: Top, centered, biggest
        title.set_color(BLACK)
        title.scale(0.9).to_edge(UP)

        # Author/Advisor/Date - All together, smaller, centered under title
        subtitles = (
            VGroup(author, university, date)
            .arrange(direction=DOWN)      # Arrange them vertically and centered
            .scale(0.8)
            .next_to(title,DOWN*3.5)
        )
        subtitles.set_color(BLACK) 

        # Get university logo
        byu_logo = SVGMobject('figures/byu_logo.svg').scale(0.8).to_edge(DOWN)
        
        # Return all objects
        return VDict([('main_title',title), ('main_subtitle',subtitles), ('logo', byu_logo)])
    

    ## ........... OVERVIEW AND MOTIVATION SLIDES ........... ##
    def overview_slides(self):
        scattering_scene = self.get_scattering_scene()
        scene = scattering_scene.scene

        # Draw axis and physical obstacles
        self.play(Write(scene['ax']))
        self.play(DrawBorderThenFill(scene['obstacles']))
        self.play(Write(scene['obstacle_labels']), Write(scene['orig_omega_label']))

        incident_wave_label = scene['wave_labels'][1].set_color(DARK_BROWN)
        scattered_wave_label = scene['wave_labels'][0].set_color(PURE_BLUE)

        # Animate incoming incident wave
        self.next_slide(loop=True)
        self.add(scene['incident_wave'], incident_wave_label)
        incident_wave_animation, scattered_wave_animation = scattering_scene.get_incident_wave_animations()
        self.play(incident_wave_animation, rate_func=linear, run_time=1)

        # Animate resulting scattered wave
        self.next_slide(loop=True)
        self.add(scene['scattered_wave'], scattered_wave_label)
        scattering_scene.reset_animation()
        self.play(incident_wave_animation, scattered_wave_animation, rate_func=linear, run_time=1)

        self.next_slide()
        return scattering_scene
    

    def motivating_equation_slides(self, scattering_scene):
        """Motivate the Helmholtz equation for this problem"""
        scene = scattering_scene.scene
        
        # Move the axis to the left 
        plot_stuff = VGroup(
            scene['ax'],
            scene['obstacles'], 
            scene['obstacle_labels'],
            scene['orig_omega_label'],
        )

        self.play(plot_stuff.animate.to_edge(RIGHT))

        # Add the Helmholtz equation to the left-hand side 
        governing_equation = MathTex(
            r'''&{} \Delta u + k^2 u = 0 \quad \text{ in } \Omega \\
                &{} \mathcal{B} u = -\mathcal{B} u_{inc} \quad \text{ on } \Gamma = \bigcup_{m=1}^{M} \Gamma_m \\
                &{} \lim_{r \to \infty} r^{1/2} ( \partial_r u - iku ) = 0''',
            font_size=40,
            color=BLACK
        ).to_edge(LEFT)
        tex_to_color_map = {
            'u': {
                'color': scattering_scene.scattered_wave_color,
                'indexes': [
                    [0,[1,5,12,44,48]]
                ] 
            },
            'u_{inc}': {
                'color': scattering_scene.incident_wave_color,
                'indexes': [
                    [0, [16,17,18,19]]
                ]
            }
        }
        governing_equation = self.color_latex(governing_equation, tex_to_color_map)
        
        self.play(TransformMatchingShapes(scene['wave_labels'], governing_equation))
        self.next_slide()

        
        # Emphasize the decomposition into u_m rather than a single u
        u_1_label = MathTex(r'u_1', font_size=40, color=PURE_BLUE).next_to(scene['obstacles'][0], RIGHT*6.1)
        u_2_label = MathTex(r'u_2', font_size=40, color=PURE_BLUE).next_to(scene['obstacles'][1], RIGHT*6.1)
        u_3_label = MathTex(r'u_3', font_size=40, color=PURE_BLUE).next_to(scene['obstacles'][2], UP*6.1)
        u_m_labels = VGroup(u_1_label, u_2_label, u_3_label)
        
        self.play(FadeOut(scene['incident_wave']), FadeOut(scene['scattered_wave'][1:]))
        self.play(Write(u_1_label))
        self.play(FadeIn(scene['scattered_wave'][1]), Write(u_2_label))
        self.play(FadeIn(scene['scattered_wave'][2]), Write(u_3_label))

        self.next_slide() 
        self.play(governing_equation.animate.to_edge(UP))
        
        decomposition_equation = MathTex(
            r'u = \sum_{m=1}^{M} u_m',
            font_size=40,
            color=BLACK
        ).next_to(governing_equation, DOWN*2).shift(LEFT)
        tex_to_color_map = {
            'u': {
                'color': scattering_scene.scattered_wave_color,
                'indexes': [
                    [0,[0]]
                ] 
            },
            'u_m': {
                'color': scattering_scene.scattered_wave_color,
                'indexes': [
                    [0, [7,8]]
                ]
            }
        }
        decomposition_equation = self.color_latex(decomposition_equation, tex_to_color_map)
        self.play(Write(decomposition_equation))

        # Finally, edit the BVP to shift into u_m notation rather than u notataion
        self.next_slide()

        new_governing_equation = MathTex(
            r'''&{} \Delta u_m + k^2 u_m = 0 \quad \text{ in } \Omega_m \\
                &{} \mathcal{B} u_m = -\mathcal{B} \left( u_{inc} + \sum_{\bar{m} \neq m} u_{\bar{m}} \right) \quad \text{ on } \Gamma_m \\
                &{} \lim_{r_m \to \infty} r_m^{1/2} ( \partial_{r_m} u_m - iku_m ) = 0 \\
                &{} (m=1,\ldots,M)''',
            font_size=35,
            color=BLACK
        ).to_edge(LEFT).shift(UP)
        tex_to_color_map = {
            r'u_m': {
                'color': scattering_scene.scattered_wave_color,
                'indexes': [
                    [0, [1,2,6,7,15,16,58,59,63,64]]
                ]
            },
            r'u_{\bar{m}}': {
                'color': DARK_BLUE,
                'indexes': [
                    [0, [33,34,35]]
                ]
            },
            r'u_{inc}': {
                'color': scattering_scene.incident_wave_color,
                'indexes': [
                    [0,[22,23,24,25]]
                ]
            }
        }
        new_governing_equation = self.color_latex(new_governing_equation, tex_to_color_map)
        old_eqs = VGroup(governing_equation, decomposition_equation)
        self.play(TransformMatchingShapes(old_eqs, new_governing_equation))

        # Now, get rid of the waves to clear the plot, and display artificial boundaries 
        self.next_slide() 

        self.play(FadeOut(u_m_labels), FadeOut(scene['scattered_wave']))

        # Recreate artificial boundaries at this new location
        scene['artificial_boundaries'] = scattering_scene._add_artificial_boundaries(scene['ax'])
        labels = scattering_scene._add_labels(scene['ax'], scene['obstacles'], scene['artificial_boundaries'])
        scene['artificial_boundary_labels'] = labels[1]
        scene['domain_labels'] = labels[2]
        self.play(DrawBorderThenFill(scene['artificial_boundaries']), Write(scene['artificial_boundary_labels']), Write(scene['domain_labels']))
        
        # FINALLY, represent this with an absorbing boundary condition at C_m:
        self.next_slide()

        final_governing_equation = MathTex(
            r'''&{} \Delta u_m + k^2 u_m = 0 \quad \text{ in } \Omega_m^- \\
                &{} \mathcal{B} u_m = -\mathcal{B} \left( u_{inc} + \sum_{\bar{m} \neq m} u_{\bar{m}} \right) \quad \text{ on } \Gamma_m \\
                &{} \operatorname{ABC}[u_m] = 0 \quad \text{ on } \mathcal{C}_m \\
                &{} (m=1,\ldots,M)''',
            font_size=35,
            color=BLACK
        ).to_edge(LEFT).shift(UP)
        tex_to_color_map = {
            r'u_m': {
                'color': scattering_scene.scattered_wave_color,
                'indexes': [
                    [0, [1,2,6,7,16,17,47,48]]
                ]
            },
            r'u_{\bar{m}}': {
                'color': DARK_BLUE,
                'indexes': [
                    [0, [34,35,36]]
                ]
            },
            r'u_{inc}': {
                'color': scattering_scene.incident_wave_color,
                'indexes': [
                    [0,[23,24,25,26]]
                ]
            }
        }
        final_governing_equation = self.color_latex(final_governing_equation, tex_to_color_map)

        abc_explanation = (
            Tex(
                'Where the operator ABC denotes an appropriate absorbing boundary condition', 
                font_size=27,
                color=BLACK
            ).next_to(final_governing_equation, DOWN*9)
            .shift(RIGHT*0.9)
        )
        self.play(FadeTransform(new_governing_equation, final_governing_equation), FadeIn(abc_explanation))

        # End of this section. Fade out everything
        self.next_slide()
        self.play(
            *[FadeOut(mob)for mob in self.mobjects]
        )

    
    ## ........... BOUNDARY CONDITION FORMULATION SLIDES ........... ##
    def karp_slides(self, scattering_scene):
        scene = scattering_scene.scene 
        self.camera.frame.save_state()

        self.play(
            FadeIn(scene['ax']),
            FadeIn(scene['obstacles']),
            FadeIn(scene['obstacle_labels']),
            FadeIn(scene['artificial_boundaries']),
            FadeIn(scene['artificial_boundary_labels']),
        )

        ax = scene['ax']
        focus_obs = scene['obstacles'][0]
        focus_obs_parametric_eq = scattering_scene.obstacle_boundaries[0]
        focus_bndry = scene['artificial_boundaries'][0]
        focus_center = scattering_scene.centers[0]
        focus_ab_radius = scattering_scene.artificial_radii[0]

        # Focus in on first obstacle and start setting up to explain Karp expansion
        camera_movement = self.camera.frame.animate.move_to(focus_obs).shift(7*RIGHT/4).set(width=6.3)  # Hard-coded, but what else do you do?
        self.play(camera_movement)
        
        title = Tex(
            'Karp Farfield Expansion (KFE) ABC',
            font_size=18,
            color=BLACK
        ).next_to(focus_bndry, RIGHT/2 + 2*UP/3)

        # Add arrows pointing outward from artificial boundary
        arrs = VGroup()
        for j in range(4):
            theta = (1+2*j)*PI/4
            start_x = focus_center[0] + focus_ab_radius * np.cos(theta) 
            start_y = focus_center[1] + focus_ab_radius * np.sin(theta) 
            end_x = focus_center[0] + (focus_ab_radius + 1) * np.cos(theta) 
            end_y = focus_center[1] + (focus_ab_radius + 1) * np.sin(theta) 

            arr = Arrow(
                start=ax.c2p(start_x, start_y),
                end=ax.c2p(end_x, end_y),
                stroke_width=2.5,
                buff=0.02
            ).set_color(PURE_RED)
            arrs.add(arr)   
        self.play(Create(arrs))

        explanation_1 = Tex(
            'Outside the circle $\mathcal{C}_m$, we can represent $u_m$ as',
            font_size=15,
            color=BLACK,
            tex_environment='flushleft',
        ).next_to(title, DOWN, aligned_edge=LEFT, buff=0.2)
        explanation_2 = Tex(
            'the following convergent ``farfield\'\' expansion:',
            font_size=15,
            color=BLACK,
            tex_environment='flushleft',
        ).next_to(explanation_1, DOWN, aligned_edge=LEFT, buff=0.1)
        
        karp_equation = MathTex(
            r'''\mathcal{K}_m (r, \theta) = H_0(kr) \sum_{l=0}^{\infty} \frac{F_l(\theta)}{(kr)^l} + H_1(kr) \sum_{l=0}^{\infty} \frac{G_l(\theta)}{(kr)^l}''',
            font_size=15,
            color=BLACK
        ).next_to(explanation_2, DOWN, aligned_edge=LEFT, buff=0.2)
        tex_to_color_map = {
            r'\mathcal{K}_m': {
                'color': PURE_RED,
                'indexes': [
                    [0, [0,1]]
                ]
            }
        }
        karp_equation = self.color_latex(karp_equation, tex_to_color_map)


        explanation_3 = Tex(
            r'where $F_l(\theta), G_l(\theta)$ satisfy',
            font_size=15,
            color=BLACK,
            tex_environment='flushleft'
        ).next_to(karp_equation, DOWN, aligned_edge=LEFT, buff=0.2)

        recursion_formulas = MathTex(
            r'''2l G_{m,l}(\theta) &= (l-1)^2 F_{m,l-1}(\theta) + F_{m,l-1}'' (\theta) \\
                2l F_{m,l}(\theta) &= -l^2 G_{m,l-1}(\theta) - G_{m,l-1}''(\theta) ''',
            font_size=15,
            color=BLACK
        ).next_to(explanation_3, DOWN, aligned_edge=LEFT, buff=0.2)
        
        karp_symbol = MathTex(
            r'\mathcal{K}_1',
            font_size=22,
            color=PURE_RED).next_to(arrs[-1], DOWN/2)
        self.play(FadeIn(title), FadeIn(explanation_1), FadeIn(explanation_2), FadeIn(karp_equation), FadeIn(karp_symbol), FadeIn(explanation_3), FadeIn(recursion_formulas))

        # Add explanation of truncation of the farfield expansion
        self.next_slide()
        
        truncation_explanation_1 = Tex(
            'Our ABC is formulated by first truncating this',
            font_size=15,
            color=BLACK,
            tex_environment='flushleft',
        ).next_to(title, DOWN, aligned_edge=LEFT, buff=0.2)
        truncation_explanation_2 = Tex(
            'expression to $L$ terms:',
            font_size=15,
            color=BLACK,
            tex_environment='flushleft',
        ).next_to(truncation_explanation_1, DOWN, aligned_edge=LEFT, buff=0.1)
        
        truncated_karp_equation = MathTex(
            r'''\mathcal{K}_{m,L} (r, \theta) = H_0(kr) \sum_{l=0}^{L-1} \frac{F_l(\theta)}{(kr)^l} + H_1(kr) \sum_{l=0}^{L-1} \frac{G_l(\theta)}{(kr)^l}''',
            font_size=14,
            color=BLACK
        ).next_to(truncation_explanation_2, DOWN, aligned_edge=LEFT, buff=0.2)
        tex_to_color_map = {
            r'\mathcal{K}_{m,L}': {
                'color': PURE_RED,
                'indexes': [
                    [0, [0,1,2,3]]
                ]
            }
        }
        truncated_karp_equation = self.color_latex(truncated_karp_equation, tex_to_color_map)
    
        truncated_karp_symbol = MathTex(
            r'\mathcal{K}_{1,L}',
            font_size=22,
            color=PURE_RED).next_to(arrs[-1], DOWN/2)
        
        self.play(FadeOut(explanation_1), FadeOut(explanation_2), FadeOut(explanation_3), FadeOut(recursion_formulas))
        self.play(
            FadeIn(truncation_explanation_1),
            FadeIn(truncation_explanation_2),
            TransformMatchingShapes(karp_equation, truncated_karp_equation),
            TransformMatchingShapes(karp_symbol, truncated_karp_symbol)
        )


        # Now, explain the boundary conditions associated with this expansion 
        self.next_slide()
        self.play(
            FadeOut(truncation_explanation_1),
            FadeOut(truncation_explanation_2),
            truncated_karp_equation.animate.next_to(title, DOWN, aligned_edge=LEFT, buff=0.2)
        )

        boundary_cond_explanation_1 = Tex(
            'Then, the following conditions are imposed at $\mathcal{C}_m$:',
            font_size=14,
            color=BLACK,
            tex_environment='flushleft',
        ).next_to(truncated_karp_equation, DOWN, aligned_edge=LEFT, buff=0.1)
        self.play(FadeIn(boundary_cond_explanation_1))

        boundary_cond_bullets = Tex(
            r'''\item Continuity of the scattered field \newline
    $\displaystyle u_m = \mathcal{K}_{m,L}$

    \item Continuity of the normal derivative: \newline
    $\displaystyle \frac{\partial u_m}{\partial r_m} = \frac{\partial \mathcal{K}_{m,L}}{\partial r_m}$

    \item Continuity of the second derivative: \newline
    $\displaystyle H_0(kR_m) \left[(L-1)^2 F_{m, L-1} + F_{m,L-1}''\right] +$ \newline 
    $\displaystyle \quad H_1(kR_m)\left[L^2 G_{m,L-1} + G_{m,L-1}''\right] = 0 $''',
            font_size=14,
            color=BLACK,
            tex_environment='itemize'
        ).next_to(boundary_cond_explanation_1, DOWN, aligned_edge=LEFT, buff=0.1)
        tex_to_color_map = {
            'u_m': {
                'color': PURE_BLUE,
                'indexes': [
                    [0, [29,30,70,71]]
                ]
            },
            r'\mathcal{K}_{m,L}': {
                'color': PURE_RED,
                'indexes': [
                    [0, [32,33,34,35,78,79,80,81]]
                ]
            }
        }
        boundary_cond_bullets = self.color_latex(boundary_cond_bullets, tex_to_color_map)

        x_sc_begin = focus_center[0] + focus_obs_parametric_eq[0](7*PI/4)
        y_sc_begin = focus_center[1] + focus_obs_parametric_eq[1](7*PI/4)
        x_sc_end = focus_center[0] + focus_ab_radius * np.cos(7*PI/4) 
        y_sc_end = focus_center[1] + focus_ab_radius * np.sin(7*PI/4) 
        u_sc_arrow = Arrow(
                start=ax.c2p(x_sc_begin, y_sc_begin),
                end=ax.c2p(x_sc_end, y_sc_end),
                stroke_width=2.5,
                buff=0.02
        ).set_color(PURE_BLUE)

        u_sc_arrow_label = (
            MathTex(
                r'u_m',font_size=22,
                color=PURE_BLUE
            ).next_to(u_sc_arrow, RIGHT/12).shift(UP/6 + LEFT/5)
        )
        
        self.play(FadeIn(boundary_cond_bullets), FadeIn(u_sc_arrow), Write(u_sc_arrow_label))

        # Now, get the influence of other obstacles on this obstacle 
        self.next_slide() 
        self.play(FadeOut(boundary_cond_explanation_1), FadeOut(boundary_cond_bullets))

        obstacle_influence_arrs = VGroup()
        obstacle_influence_labels = VGroup()
        coords = [(None,None), (2.2,2), (0.8,3.3)]
        for m,(obstacle, lbl_coords) in enumerate(zip(scene['obstacles'], coords)):
            if m != 0:
                arr = Arrow(obstacle, focus_obs, stroke_width=2.5, max_tip_length_to_length_ratio=0.05, buff=0.02, color=DARK_BLUE)
                obstacle_influence_arrs.add(arr)

                lbl = MathTex(rf'\mathcal{{K}}_{{{m+1},L}}', font_size=22, color=DARK_BLUE).move_to(ax.c2p(*lbl_coords))
                obstacle_influence_labels.add(lbl)

        obstacle_influence_explanation_1 = (
            Tex(
                'We can represent influences of other obstacles',
                font_size=14,
                color=BLACK,
                tex_environment='flushleft',
            ).next_to(truncated_karp_equation, DOWN, aligned_edge=LEFT, buff=0.5)
        )
        obstacle_influence_explanation_2 = (
            Tex(
                r'by using their truncated Karp expansions $\mathcal{K}_{\bar{m}, L}$',
                font_size=14,
                color=BLACK,
                tex_environment='flushleft',
            ).next_to(obstacle_influence_explanation_1, DOWN, aligned_edge=LEFT, buff=0.1)
        )
        tex_to_color_map = {
            r'\mathcal{K}_{\bar{m}, L}' : {
                'color': DARK_BLUE,
                'indexes': [
                    [0,[35,36,37,38,39]]
                ]
            }
        }
        obstacle_influence_explanation_2 = self.color_latex(obstacle_influence_explanation_2, tex_to_color_map)
        self.play(FadeIn(obstacle_influence_arrs), Write(obstacle_influence_labels), FadeIn(obstacle_influence_explanation_1), FadeIn(obstacle_influence_explanation_2))

        # Now, display the final BVP
        self.next_slide()
        self.play(
            FadeOut(obstacle_influence_explanation_1),
            FadeOut(obstacle_influence_explanation_2),
            FadeOut(title),
            FadeOut(truncated_karp_equation)
        )

        final_bvp_title = Tex(
            'Final BVP System',
            font_size=18,
            color=BLACK
        ).next_to(focus_bndry, RIGHT/2 + 2*UP/3)

        bvp = (
            MathTex(
                r'''
                & \Delta u_m + k^2 u_m = 0 \quad \text{ in } \Omega_m^{-} \\
                & \mathcal{B}_m u_m = -\mathcal{B}_m \left(u_{inc} + \sum_{\bar{m} \neq m} \mathcal{K}_{\bar{m}, L}\right) \quad \text{ on } \Gamma_m \\
                & u_m = \mathcal{K}_{m,L} \quad \text{ on } \mathcal{C}_m \\
                & \frac{\partial u_m}{\partial r_m} = \frac{\partial \mathcal{K}_{m,L}}{\partial r_m} \quad \text{ on } \mathcal{C}_m \\
                & H_0(kR_m) \left[(L-1)^2 F_{m, L-1} + F_{m,L-1}''\right] + \\
                &\quad H_1(kR_m)\left[L^2 G_{m,L-1} + G_{m,L-1}''\right] = 0 \quad \text{ on } \mathcal{C}_m \\
                &2l G_{m,l}(\theta) = (l-1)^2 F_{m,l-1}(\theta) + F_{m,l-1}''(\theta) \\
                &2l F_{m,l}(\theta) = -l^2 G_{m,l-1}(\theta) - G_{m,l-1}''(\theta)
                ''',
                font_size=14,
                color=BLACK
            ).next_to(final_bvp_title, DOWN, aligned_edge=LEFT, buff=0.1)
        )
        tex_to_color_map = {
            r'u_m': {
                'color': PURE_BLUE,
                'indexes': [
                    [0,[1,2,6,7,17,18,47,48,59,60]]
                ]

            },
            r'u_{inc}': {
                'color': DARK_BROWN,
                'indexes': [
                    [0,[25,26,27,28]]
                ]

            },
            r'\mathcal{K}_{\bar{m}, L}': {
                'color': DARK_BLUE,
                'indexes': [
                    [0,[36,37,38,39,40]]
                ]

            },
            r'\mathcal{K}_{m, L}': {
                'color': PURE_RED,
                'indexes': [
                    [0,[50,51,52,53,67,68,69,70]]
                ]
            }
        }
        bvp = self.color_latex(bvp, tex_to_color_map)
        self.play(FadeIn(final_bvp_title, bvp))

        self.next_slide()
        self.play(
            *[FadeOut(mob)for mob in self.mobjects]
        )
        self.play(Restore(self.camera.frame))   # Reset camera frame

    
    ## ............ RESIDUAL METHOD FORMULATION SLIDES ............. ##
    def boundary_guess_slides(self):
        # Get scattering scene for future display
        scattering_scene = self.get_scattering_scene()
        scene = scattering_scene.scene
        

        # Add the scattering scene to the right of the screen again
        plot_stuff = VDict(
            {
                'ax': scene['ax'],
                'obstacles': scene['obstacles'],
                'obstacle_labels': scene['obstacle_labels'],
                'artificial_boundaries': scene['artificial_boundaries'],
                'artificial_boundary_labels': scene['artificial_boundary_labels']
            }
        ).to_edge(RIGHT)

        # Add the long form of the scattering BVP
        bvp = (
            MathTex(
                r'''
                & \Delta u_m + k^2 u_m = 0 \quad \text{ in } \Omega_m^{-} \\
                & \mathcal{B}_m u_m = -\mathcal{B}_m \left(u_{inc} + \sum_{\bar{m} \neq m} \mathcal{K}_{\bar{m}, L}\right) \quad \text{ on } \Gamma_m \\
                & u_m = \mathcal{K}_{m,L} \quad \text{ on } \mathcal{C}_m \\
                & \frac{\partial u_m}{\partial r_m} = \frac{\partial \mathcal{K}_{m,L}}{\partial r_m} \quad \text{ on } \mathcal{C}_m \\
                & H_0(kR_m) \left[(L-1)^2 F_{m, L-1} + F_{m,L-1}''\right] + \\
                &\quad H_1(kR_m)\left[L^2 G_{m,L-1} + G_{m,L-1}''\right] = 0 \quad \text{ on } \mathcal{C}_m \\
                &2l G_{m,l}(\theta) = (l-1)^2 F_{m,l-1}(\theta) + F_{m,l-1}''(\theta) \\
                &2l F_{m,l}(\theta) = -l^2 G_{m,l-1}(\theta) - G_{m,l-1}''(\theta)
                ''',
                font_size=30,
                color=BLACK
            ).to_edge(LEFT)
            .shift(UP*0.5)
        )
        tex_to_color_map = {
            r'u_m': {
                'color': PURE_BLUE,
                'indexes': [
                    [0,[1,2,6,7,17,18,47,48,59,60]]
                ]

            },
            r'u_{inc}': {
                'color': DARK_BROWN,
                'indexes': [
                    [0,[25,26,27,28]]
                ]

            },
            r'\mathcal{K}_{\bar{m}, L}': {
                'color': DARK_BLUE,
                'indexes': [
                    [0,[36,37,38,39,40]]
                ]

            },
            r'\mathcal{K}_{m, L}': {
                'color': PURE_RED,
                'indexes': [
                    [0,[50,51,52,53,67,68,69,70]]
                ]
            }
        }
        bvp = self.color_latex(bvp, tex_to_color_map)

        self.play(FadeIn(plot_stuff), FadeIn(bvp))

        # Transform to easier-to-read version
        self.next_slide()
        new_bvp = (
            MathTex(
                r'''
                & \Delta u_m + k^2 u_m = 0 \quad \text{ in } \Omega_m^{-} \\
                & \mathcal{B}_m u_m = -\mathcal{B}_m \left(u_{inc} + \sum_{\bar{m} \neq m} \mathcal{K}_{\bar{m}, L}\right) \quad \text{ on } \Gamma_m \\
                & \operatorname{MKFE}[u_m, \mathcal{K}_{m, L}] = 0 \quad \text{ on } \mathcal{C}_m
                ''',
                font_size=30,
                color=BLACK
            ).to_edge(LEFT)
            .shift(UP*2)
        )
        tex_to_color_map = {
            r'u_m': {
                'color': PURE_BLUE,
                'indexes': [
                    [0,[1,2,6,7,17,18,52,53]]
                ]

            },
            r'u_{inc}': {
                'color': DARK_BROWN,
                'indexes': [
                    [0,[25,26,27,28]]
                ]

            },
            r'\mathcal{K}_{\bar{m}, L}': {
                'color': DARK_BLUE,
                'indexes': [
                    [0,[36,37,38,39,40]]
                ]
            },
            r'\mathcal{K}_{m, L}': {
                'color': PURE_RED,
                'indexes': [
                    [0,[55,56,57,58]]
                ]
            }
        }
        new_bvp = self.color_latex(new_bvp, tex_to_color_map)
        self.play(TransformMatchingShapes(bvp, new_bvp))

        # Now, add description of how to decouple the problem 
        self.next_slide()
        decoupling_description_1 = (
            Tex(
                r'We can replace the boundary data on $\Gamma_m$',
                font_size=30,
                color=BLACK
            ).next_to(new_bvp, DOWN, aligned_edge=LEFT, buff=0.5)
        )
        decoupling_description_2 = (
            Tex(
                r'with a guess $\psi_m$.',
                font_size=30,
                color=BLACK
            ).next_to(decoupling_description_1, DOWN, aligned_edge=LEFT, buff=0.1)
        )
        tex_to_color_map = {
            r'psi_m': {
                'color': self.PSI_COLOR,
                'indexes': [
                    [0,[10,11,12]]
                ]

            }
        }
        decoupling_description_2 = self.color_latex(decoupling_description_2, tex_to_color_map)

        decoupling_descriptions = VGroup(decoupling_description_1, decoupling_description_2)

        # Replace the boundary conditions appropriately
        bvp_with_guess = (
            MathTex(
                r'''
                & \Delta u_m + k^2 u_m = 0 \quad \text{ in } \Omega_m^{-} \\
                & \mathcal{B}_m u_m = \psi_m \quad \text{ on } \Gamma_m \\
                & \operatorname{MKFE}[u_m, \mathcal{K}_{m, L}] = 0 \quad \text{ on } \mathcal{C}_m
                ''',
                font_size=40,
                color=BLACK
            ).move_to(new_bvp)
        )
        tex_to_color_map = {
            r'u_m': {
                'color': PURE_BLUE,
                'indexes': [
                    [0,[1,2,6,7,17,18,31,32]]
                ]
            },
            r'\psi_m': {
                'color': self.PSI_COLOR,
                'indexes': [
                    [0,[20,21]]
                ]
            },
            r'\mathcal{K}_{m, L}': {
                'color': PURE_RED,
                'indexes': [
                    [0,[34,35,36,37]]
                ]
            }
        }
        bvp_with_guess = self.color_latex(bvp_with_guess, tex_to_color_map)

        # Also, highlight the boundaries of the scattering problem
        psi_labels = VGroup()
        for m, obstacle in enumerate(scene['obstacles']):
            lbl = (
                MathTex(
                    rf'\psi_{m+1}',
                    font_size=30,
                    color=self.PSI_COLOR,
                ).next_to(obstacle, RIGHT*0.5)
            )
            psi_labels.add(lbl)
            
        self.play(
            FadeIn(decoupling_descriptions),
            *[obstacle.animate.set_stroke_color(self.PSI_COLOR).set_stroke_width(5) for obstacle in scene['obstacles']],
            FadeIn(psi_labels),
            TransformMatchingShapes(new_bvp, bvp_with_guess)
        )

        # Show that this allows us to decouple the problem
        self.next_slide()
        decoupling_description_3 = (
            Tex(
                r'Notice that the BVP for obstacle $m$',
                font_size=30,
                color=BLACK
            ).next_to(decoupling_description_2, DOWN, aligned_edge=LEFT, buff=0.3)
        )
        decoupling_description_4 = (
            Tex(
                r'no longer has any explicit coupling to',
                font_size=30,
                color=BLACK
            ).next_to(decoupling_description_3, DOWN, aligned_edge=LEFT, buff=0.1)
        )
        decoupling_description_5 = (
            Tex(
                r'the BVPs for the other obstacles.',
                font_size=30,
                color=BLACK
            ).next_to(decoupling_description_4, DOWN, aligned_edge=LEFT, buff=0.1)
        )
        
        decoupling_descriptions_ii = VGroup(
            decoupling_description_3, 
            decoupling_description_4, 
            decoupling_description_5,
        )

        
        self.next_slide()
        decoupling_description_6 = (
             Tex(
                r'Also, if $\displaystyle \psi_m = -\mathcal{B}_m \left(u_{inc} + \sum_{\bar{m} \neq m} \mathcal{K}_{\bar{m}, L}\right)$,',
                font_size=30,
                color=BLACK
            ).next_to(decoupling_description_5, DOWN, aligned_edge=LEFT, buff=0.3)
        )
        tex_to_color_map = {
            r'\psi_m': {
                'color': self.PSI_COLOR,
                'indexes': [
                    [0,[7,8]]
                ]

            },
            r'u_{inc}': {
                'color': DARK_BROWN,
                'indexes': [
                    [0,[15,16,17,18]]
                ]

            },
            r'\mathcal{K}_{\bar{m}, L}': {
                'color': DARK_BLUE,
                'indexes': [
                    [0,[26,27,28,29,30]]
                ]
            },
        }
        decoupling_description_6 = self.color_latex(decoupling_description_6, tex_to_color_map)

        decoupling_description_7 = (
             Tex(
                r'the solution of the BVP will be correct.',
                font_size=30,
                color=BLACK
            ).next_to(decoupling_description_6, DOWN, aligned_edge=LEFT, buff=0.1)
        )
        
        decoupling_descriptions_iii = VGroup(
            decoupling_description_6, 
            decoupling_description_7
        )
        self.play(FadeIn(decoupling_descriptions_ii), FadeIn(decoupling_descriptions_iii))

        # Add description about how iterative method.
        self.next_slide()
        self.play(
            FadeOut(decoupling_descriptions),
            FadeOut(decoupling_descriptions_ii),
            FadeOut(decoupling_descriptions_iii)
        )

        iterative_method_intro = (
            Tex(
                r'We can create an iterative method by:',
                font_size=30,
                color=BLACK
            ).next_to(bvp_with_guess, DOWN, aligned_edge=LEFT, buff=0.8)
        )
        self.play(FadeIn(iterative_method_intro))
        
        iterative_method_bullets = (
            Tex(
                r'\item Giving initial guesses $\psi_m^{(0)}$ at $\Gamma_m$',
                r'\item Solving the BVP with these guesses',
                r'\item Updating guesses appropriately to get $\psi_m^{(1)}$',
                r'\item Continuing until convergence',
                tex_environment='enumerate',
                font_size=30,
                color=BLACK
            ).next_to(iterative_method_intro, DOWN, aligned_edge=LEFT, buff=0.5)
        )
        tex_to_color_map = {
            r'\psi_m^{(*)}': {
                'color': self.PSI_COLOR,
                'indexes': [
                    [0,[22,23,24,25,26]],
                    [2,[35,36,37,38,39]]
                ]
            }
        }
        iterative_method_bullets = self.color_latex(iterative_method_bullets, tex_to_color_map)

        # FIRST STEP - INITIAL GUESS
        self.next_slide()
        animations = []

        # Update psi_m to be psi_m^{(0)} at all occurances in BVP and in diagram
        new_labels = VGroup()
        for m, obstacle in enumerate(scene['obstacles']):
            new_lbl = (
                MathTex(
                    rf'\psi_{m+1}^{{(0)}}',
                    font_size=30,
                    color=self.PSI_COLOR,
                ).next_to(obstacle, RIGHT*0.5)
            )
            new_labels.add(new_lbl)
            animations.append(TransformMatchingShapes(psi_labels[m], new_lbl))

        bvp_with_iterative_guess = (
            MathTex(
                r'''
                & \Delta u_m + k^2 u_m = 0 \quad \text{ in } \Omega_m^{-} \\
                & \mathcal{B}_m u_m = \psi_m^{(0)} \quad \text{ on } \Gamma_m \\
                & \operatorname{MKFE}[u_m, \mathcal{K}_{m, L}] = 0 \quad \text{ on } \mathcal{C}_m
                ''',
                font_size=40,
                color=BLACK
            ).move_to(bvp_with_guess, aligned_edge=LEFT)
        )
        tex_to_color_map = {
            r'u_m': {
                'color': PURE_BLUE,
                'indexes': [
                    [0,[1,2,6,7,17,18,34,35]]
                ]
            },
            r'\psi_m^{*}': {
                'color': self.PSI_COLOR,
                'indexes': [
                    [0,[20,21,22,23,24]]
                ]
            },
            r'\mathcal{K}_{m, L}': {
                'color': PURE_RED,
                'indexes': [
                    [0,[37,38,39,40]]
                ]
            }
        }
        bvp_with_iterative_guess = self.color_latex(bvp_with_iterative_guess, tex_to_color_map)
        
        
        self.play(
            *animations,
            FadeIn(iterative_method_bullets[0]),
            TransformMatchingShapes(bvp_with_guess, bvp_with_iterative_guess)
        )

        # SECOND STEP - SOLVE BVP WITH THESE GUESSES 
        self.next_slide()
        
        bvp_with_iterative_guess_ii = (
            MathTex(
                r'''
                & \Delta u_m^{(0)} + k^2 u_m^{(0)} = 0 \quad \text{ in } \Omega_m^{-} \\
                & \mathcal{B}_m u_m^{(0)} = \psi_m^{(0)} \quad \text{ on } \Gamma_m \\
                & \operatorname{MKFE}[u_m^{(0)}, \mathcal{K}_{m, L}^{(0)}] = 0 \quad \text{ on } \mathcal{C}_m
                ''',
                font_size=40,
                color=BLACK
            ).move_to(bvp_with_iterative_guess, aligned_edge=LEFT)
        )
        tex_to_color_map = {
            r'u_m^{*}': {
                'color': PURE_BLUE,
                'indexes': [
                    [0,[1,2,3,4,5, 9,10,11,12,13,23,24,25,26,27,43,44,45,46,47]]
                ]
            },
            r'\psi_m^{*}': {
                'color': self.PSI_COLOR,
                'indexes': [
                    [0,[29,30,31,32,33]]
                ]
            },
            r'\mathcal{K}_{m, L}^{*}': {
                'color': PURE_RED,
                'indexes': [
                    [0,[49,50,51,52,53,54,55]]
                ]
            }
        }
        bvp_with_iterative_guess_ii = self.color_latex(bvp_with_iterative_guess_ii, tex_to_color_map)
        
        self.play(FadeIn(iterative_method_bullets[1]), TransformMatchingShapes(bvp_with_iterative_guess, bvp_with_iterative_guess_ii))

        # THIRD STEP - UPDATE GUESSES APPROPRIATELY TO GET NEW BOUNDARY GUESSES 
        self.next_slide()

        new_labels_ii = VGroup()
        for m, obstacle in enumerate(scene['obstacles']):
            new_lbl = (
                MathTex(
                    rf'\psi_{m+1}^{{(1)}}',
                    font_size=30,
                    color=self.PSI_COLOR,
                ).next_to(obstacle, RIGHT*0.5)
            )
            new_labels_ii.add(new_lbl)
            animations.append(TransformMatchingShapes(new_labels[m], new_lbl))

        self.play(*animations, FadeIn(iterative_method_bullets[2]))

        # FINAL STEP - REPEAT UNTIL CONVERGENCE 
        self.next_slide()
        
        bvp_with_iterative_guess_iii = (
            MathTex(
                r'''
                & \Delta u_m^{(1)} + k^2 u_m^{(1)} = 0 \quad \text{ in } \Omega_m^{-} \\
                & \mathcal{B}_m u_m^{(1)} = \psi_m^{(1)} \quad \text{ on } \Gamma_m \\
                & \operatorname{MKFE}[u_m^{(1)}, \mathcal{K}_{m, L}^{(1)}] = 0 \quad \text{ on } \mathcal{C}_m
                ''',
                font_size=40,
                color=BLACK
            ).move_to(bvp_with_iterative_guess_ii, aligned_edge=LEFT)
        )
        tex_to_color_map = {
            r'u_m^{*}': {
                'color': PURE_BLUE,
                'indexes': [
                    [0,[1,2,3,4,5, 9,10,11,12,13,23,24,25,26,27,43,44,45,46,47]]
                ]
            },
            r'\psi_m^{*}': {
                'color': self.PSI_COLOR,
                'indexes': [
                    [0,[29,30,31,32,33]]
                ]
            },
            r'\mathcal{K}_{m, L}^{*}': {
                'color': PURE_RED,
                'indexes': [
                    [0,[49,50,51,52,53,54,55]]
                ]
            }
        }
        bvp_with_iterative_guess_iii = self.color_latex(bvp_with_iterative_guess_iii, tex_to_color_map)
        
        self.play(TransformMatchingShapes(bvp_with_iterative_guess_ii, bvp_with_iterative_guess_iii), FadeIn(iterative_method_bullets[3]))
        
        # Remove everything on the slide
        self.next_slide()
        self.play(
            *[FadeOut(mob)for mob in self.mobjects]
        )

        next_slide_items = VDict( 
            {
                'plot_stuff': plot_stuff,
                'initial_bvp': bvp_with_iterative_guess_ii,
                'initial_labels': new_labels
            }
        )
        return next_slide_items


    def residual_method_slides(self, prev_slide_items):
        ### GET CONSTANTS 
        ax = prev_slide_items['plot_stuff']['ax']
        obstacles = prev_slide_items['plot_stuff']['obstacles']
        initial_labels = prev_slide_items['initial_labels']

        # Introduce GMRES Method
        residual_method_title = (
            Tex(
                r'Minimal-Residual (GMRES) Method',
                font_size=45,
                color=BLACK
            ).to_edge(UP)
        )
        
        paper_reference = (
            Tex(
                r'\mbox{[1] Z. Xie, R. Zhang, B. Wang, L.-L. Wang, An efficient iterative method for solving multiple scattering in locally inhomogeneous media,}\newline',
                r'\quad \quad \quad \quad Computer Methods in Applied Mechanics and Engineering 358 (2020) 112642.',
                font_size=18,
                color=BLACK
            ).to_edge(DOWN).to_edge(LEFT)
        )

        intro_text = (
           Tex(
                r'\item Similar to the approach used by Xie, Zhang, et al. [1], we use a GMRES-based approach to solve this BVP',
                r'\item \textbf{Main Idea}: Use error residual vectors as boundary forcing terms',
                r'\item Recover a sequence of boundary data $(\psi_m^{(n)})_{n \in \mathbb{N}}$ '
                r' such that'
                r' $$\left \lVert -\mathcal{B}(u_{inc}) - \psi^{(n)}\right \rVert$$ '
                r' where $\mathcal{B}(u_{inc}) = \begin{bmatrix}\mathcal{B}_1(u_{inc}) & \ldots & \mathcal{B}_M(u_{inc}) \end{bmatrix}^\intercal$ \newline'
                r' and $\psi^{(n)} = \begin{bmatrix}\psi_1^{(n)} & \ldots & \psi_M^{(n)} \end{bmatrix}^\intercal$',
                tex_environment='itemize',
                font_size=30,
                color=BLACK
            ).next_to(residual_method_title, DOWN, buff=0.4)
        )
        tex_to_color_map = {
            r'\psi_m^{(*)}': {
                'color': self.PSI_COLOR,
                'indexes': [
                    [2,[32,33,34,35,36,61,62,63,64,112,113,114,115,116,120,121,122,123,124]]
                ]
            },
            r'u_inc': {
                'color': DARK_BROWN,
                'indexes': [
                    [2, [55,56,57,58,75,76,77,78,85,86,87,88,106,107,108,109]]
                ]
            }
        }
        intro_text = self.color_latex(intro_text, tex_to_color_map)
        
        self.play(FadeIn(residual_method_title), FadeIn(paper_reference), FadeIn(intro_text))

        
        # Now, let's demonstrate this visually 
        self.next_slide()
        self.play(FadeOut(residual_method_title), FadeOut(paper_reference), FadeOut(intro_text))

        # Fade in demo plot
        demo_plot = VGroup(prev_slide_items['plot_stuff'], prev_slide_items['initial_labels'])
        self.play(FadeIn(demo_plot))

        # STEP 1: Solve initial BVP
        step_1 = (
            Tex(
                r'\item \textbf{Step 1:} Given initial boundary data \newline'
                r' guesses $\psi_m^{(0)}$, solve the \newline'
                r' $M$ single scattering BVPs:',
                tex_environment='itemize',
                color=BLACK,
                font_size=34
            ).to_edge(LEFT)
            .to_edge(UP)
        )
        tex_to_color_map = {
            r'\psi_m^{*}': {
                'color': self.PSI_COLOR,
                'indexes': [
                    [0,[38,39,40,41,42]]
                ]
            }
        }
        step_1 = self.color_latex(step_1, tex_to_color_map)
        step_1_bvp = prev_slide_items['initial_bvp'].next_to(step_1, DOWN, aligned_edge=LEFT, buff=0.4)
        self.play(FadeIn(step_1), FadeIn(step_1_bvp))

        # STEP 2: Solve initial BVP
        self.next_slide()
        self.play(FadeOut(step_1), step_1_bvp.animate.to_edge(UP))

        step_2 = (
            Tex(
                r'\item \textbf{Step 2:} Compute initial residuals:',
                tex_environment='itemize',
                color=BLACK,
                font_size=34
            ).next_to(step_1_bvp, DOWN, aligned_edge=LEFT, buff=0.4)
        )

        initial_residuals = (
            MathTex(
                r'r_m^{(0)} := -\mathcal{B}_m \left(u_{inc} + \sum_{\bar{m} \neq m} \mathcal{K}_{\bar{m},L}^{(0)} \right) - \psi_m^{(0)}',
                color=BLACK,
                font_size=33
            ).next_to(step_2, DOWN, aligned_edge=LEFT, buff=0.4)
        )
        tex_to_color_map = {
            r'r_m^{*}': {
                'color': self.RESIDUAL_COLOR,
                'indexes': [
                    [0, [0,1,2,3,4]]
                ]
            },
            r'\psi_m^{*}': {
                'color': self.PSI_COLOR,
                'indexes': [
                    [0,[34,35,36,37,38]]
                ]
            },
            r'u_{inc}': {
                'color': DARK_BROWN,
                'indexes': [
                    [0,[12,13,14,15]]
                ]
            },
            r'\mathcal{K}_{\bar{m},L}^{*}': {
                'color': PURE_RED,
                'indexes': [
                    [0, [23,24,25,26,27,28,29,30]]
                ]
            }
        }
        initial_residuals = self.color_latex(initial_residuals, tex_to_color_map)

        # Draw influence of obstacles 2 and 3 on obstalce 1 with Karp expansion arrows
        influence_arrows = VGroup()
        influence_labels = VGroup()
        label_coords = [(0,0),(3.7,0.9), (-0.6,1.5)]  # HARDCODED
        for m, (obstacle, label_coord) in enumerate(zip(obstacles, label_coords)):
            if m != 0:
                arr = Arrow(
                    obstacle,
                    obstacles[0],
                    stroke_width=4.5,
                    color=PURE_RED,
                    buff=0.02
                )
                influence_arrows.add(arr)

                lbl = (
                    MathTex(
                        rf'\mathcal{{K}}_{{ {m+1},L }}^{{(0)}}',
                        color=PURE_RED,
                        font_size=30
                    ).move_to(ax.c2p(*label_coord))
                )
                influence_labels.add(lbl)

        # Draw influence of incident wave (for first iteration) ## HARDCODED
        u_inc_arrow = Arrow(
            ax.c2p(-5,6),
            obstacles[0],
            stroke_width=4.5,
            color=DARK_BROWN,
            buff=0.02
        )
        u_inc_arrow_label = (
            MathTex(
                r'u_{inc}',
                color=DARK_BROWN,
                font_size=30
            ).move_to(ax.c2p(-2,5.5))
        )

        self.play(
            FadeIn(step_2),
            FadeIn(initial_residuals),
            FadeIn(influence_labels),
            FadeIn(u_inc_arrow_label),
            *[GrowArrow(arr) for arr in influence_arrows],
            GrowArrow(u_inc_arrow)
        )

        # STEP 3: Normalize these residuals
        self.next_slide()
        self.play(
            FadeOut(influence_labels),
            FadeOut(influence_arrows),
            FadeOut(u_inc_arrow_label),
            FadeOut(u_inc_arrow),
            FadeOut(step_2),
            FadeOut(step_1_bvp),
            FadeOut(initial_residuals)
        )

        step_3 = (
            Tex(
                r'\item \textbf{Step 3:} Create and normalize the \newline combined residual:',
                tex_environment='itemize',
                font_size=34,
                color=BLACK
            ).to_edge(LEFT)
            .to_edge(UP)
        )

        stacked_residual_equation = (
            MathTex(
                r'r^{(0)} = \begin{bmatrix} r_1^{(0)} & \ldots & r_M^{(0)} \end{bmatrix}^\intercal',
                font_size=32,
                color=BLACK
            ).next_to(step_3, DOWN, aligned_edge=LEFT, buff=0.2)
        )
        tex_to_color_map = {
            r'\r_m^{*}': {
                'color': self.RESIDUAL_COLOR,
                'indexes': [
                    [0,[6,7,8,9,10,14,15,16,17,18]]
                ]
            }
        }
        stacked_residual_equation = self.color_latex(stacked_residual_equation, tex_to_color_map)

        normed_residual_equation = (
            MathTex(
                r'q^{(1)} = \frac{r^{(0)}}{\lVert r^{(0)}\rVert} := \begin{bmatrix} q_1^{(1)} & \ldots & q_M^{(1)} \end{bmatrix}^\intercal',
                font_size=32,
                color=BLACK
            ).next_to(stacked_residual_equation, DOWN, aligned_edge=LEFT, buff=0.2)
        )
        tex_to_color_map = {
            r'q^{*}': {
                'color': self.NORMED_RESIDUAL_COLOR,
                'indexes': [
                    [0,[0,1,2,3,19,20,21,22,23,27,28,29,30,31]]
                ]
            }
        }
        normed_residual_equation = self.color_latex(normed_residual_equation, tex_to_color_map)

        storage_explanation = (
            Tex(
                r'Store the result as a column \newline in a matrix $Q$:',
                color=BLACK,
                font_size=34,
                tex_environment='flushleft'
            ).next_to(normed_residual_equation, DOWN, aligned_edge=LEFT, buff=0.5)
        )

        storage_tex = (
            MathTex(
                r'Q = [q^{(1)}]',
                color=BLACK,
                font_size=34
            ).next_to(storage_explanation, DOWN, aligned_edge=LEFT, buff=0.2)
        )
        tex_to_color_map = {
        r'q^{*}': {
                'color': self.NORMED_RESIDUAL_COLOR,
                'indexes': [
                    [0,[3,4,5,6]]
                ]
            }
        }
        storage_tex = self.color_latex(storage_tex, tex_to_color_map)

        self.play(
            FadeIn(step_3),
            FadeIn(stacked_residual_equation),
            FadeIn(normed_residual_equation),
            FadeIn(storage_explanation),
            FadeIn(storage_tex)
        )

        

        # # --- STEP 4: Restart the process, but reset the boundary data with the residuals 
        # at the last iteration
        self.next_slide()

        movement_of_Q = storage_tex.animate.to_edge(LEFT).to_edge(UP)
        new_labels = VGroup()
        for m, obstacle in enumerate(obstacles):
            new_lbl = MathTex(
                rf'q_{m+1}^{{(1)}}',
                color=self.NORMED_RESIDUAL_COLOR,
                font_size=30
            ).next_to(obstacle, RIGHT*0.5)
            new_labels.add(new_lbl)    
        self.play(
            movement_of_Q,
            FadeOut(step_3),
            FadeOut(stacked_residual_equation),
            FadeOut(normed_residual_equation),
            FadeOut(storage_explanation),
            obstacles.animate.set(stroke_color=self.NORMED_RESIDUAL_COLOR),
            FadeTransform(initial_labels, new_labels),
        )
        
        step_4 = (
            Tex(
                r'\item \textbf{Step 4:} Now, solve the single-scattering BVPs, \newline'
                r' but with $q_{m}^{(1)}$ as boundary data on $\Gamma_m$:',
                color=BLACK,
                font_size=34,
                tex_environment='itemize'
            ).next_to(storage_tex, DOWN, aligned_edge=LEFT, buff=0.4)
        )
        tex_to_color_map = {
            r'q_m^{*}': {
                'color': self.NORMED_RESIDUAL_COLOR,
                'indexes': [
                    [0,[48,49,50,51,52]]
                ]
            }
        }
        step_4 = self.color_latex(step_4, tex_to_color_map)

        new_bvp = (
            MathTex(
                r'''
                & \Delta u_m^{(1)} + k^2 u_m^{(1)} = 0 \quad \text{ in } \Omega_m^{-} \\
                & \mathcal{B}_m u_m^{(1)} = q_m^{(1)} \quad \text{ on } \Gamma_m \\
                & \operatorname{MKFE}[u_m^{(1)}, \mathcal{K}_{m, L}^{(1)}] = 0 \quad \text{ on } \mathcal{C}_m
                ''',
                font_size=35,
                color=BLACK
            ).next_to(step_4, DOWN, aligned_edge=LEFT, buff=0.2)
        )
        tex_to_color_map = {
            r'u_m^{*}': {
                'color': PURE_BLUE,
                'indexes': [
                    [0,[1,2,3,4,5, 9,10,11,12,13,23,24,25,26,27,43,44,45,46,47]]
                ]
            },
            r'q_m^{*}': {
                'color': self.NORMED_RESIDUAL_COLOR,
                'indexes': [
                    [0,[29,30,31,32,33]]
                ]
            },
            r'\mathcal{K}_{m, L}^{*}': {
                'color': PURE_RED,
                'indexes': [
                    [0,[49,50,51,52,53,54,55]]
                ]
            }
        }
        new_bvp = self.color_latex(new_bvp, tex_to_color_map)
        
        self.play(FadeIn(step_4), FadeIn(new_bvp))

        # STEP 5: Compute raw residual at this iteration
        self.next_slide()
        self.play(
            FadeOut(step_4),
            new_bvp.animate.next_to(storage_tex, DOWN, aligned_edge=LEFT, buff=0.4)
        )

        step_5 = (
            Tex(
                r'\item \textbf{Step 5:} Compute new raw residuals:',
                tex_environment='itemize',
                color=BLACK,
                font_size=34
            ).next_to(new_bvp, DOWN, aligned_edge=LEFT, buff=0.4)
        )

        raw_residuals = (
            MathTex(
                r'r_m^{(1)} := \mathcal{B}_m \left(\sum_{\bar{m} \neq m} \mathcal{K}_{\bar{m},L}^{(1)} \right) + q_m^{(0)}',
                color=BLACK,
                font_size=33
            ).next_to(step_5, DOWN, aligned_edge=LEFT, buff=0.4)
        )
        tex_to_color_map = {
            r'r_m^{*}': {
                'color': self.RESIDUAL_COLOR,
                'indexes': [
                    [0, [0,1,2,3,4]]
                ]
            },
            r'\q_m^{*}': {
                'color': self.NORMED_RESIDUAL_COLOR,
                'indexes': [
                    [0,[28,29,30,31,32]]
                ]
            },
            r'\mathcal{K}_{\bar{m},L}^{*}': {
                'color': PURE_RED,
                'indexes': [
                    [0, [17,18,19,20,21,22,23,24]]
                ]
            }
        }
        raw_residuals = self.color_latex(raw_residuals, tex_to_color_map)

        # Draw influence of obstacles 2 and 3 on obstalce 1 with Karp expansion arrows
        influence_arrows = VGroup()
        influence_labels = VGroup()
        label_coords = [(0,0),(3.7,0.9), (-0.6,1.5)]  # HARDCODED
        for m, (obstacle, label_coord) in enumerate(zip(obstacles, label_coords)):
            if m != 0:
                arr = Arrow(
                    obstacle,
                    obstacles[0],
                    stroke_width=4.5,
                    color=PURE_RED,
                    buff=0.02
                )
                influence_arrows.add(arr)

                lbl = (
                    MathTex(
                        rf'\mathcal{{K}}_{{ {m+1},L }}^{{(1)}}',
                        color=PURE_RED,
                        font_size=30
                    ).move_to(ax.c2p(*label_coord))
                )
                influence_labels.add(lbl)

        self.play(
            FadeIn(step_5),
            FadeIn(raw_residuals),
            FadeIn(influence_labels),
            *[GrowArrow(arr) for arr in influence_arrows],
        )

        # STEP 6: Get orthonormal residual to all columns of Q
        self.next_slide()
        self.play(
            FadeOut(influence_labels),
            FadeOut(influence_arrows),
            FadeOut(step_5),
            FadeOut(new_bvp),
            FadeOut(raw_residuals)
        )

        step_6 = (
            Tex(
                r'\item \textbf{Step 6:} Use Arnoldi iteration \newline'
                r' to make $r^{(1)}$ orthonormal to $q^{(1)}$. \newline'
                r' Call the result $q^{(2)}$.',
                tex_environment='itemize',
                font_size=34,
                color=BLACK
            ).to_edge(LEFT).to_edge(UP)
        )
        tex_to_color_map = {
            r'\q_m^{*}': {
                'color': self.NORMED_RESIDUAL_COLOR,
                'indexes': [
                    [0,[49,50,51,52,67,68,69,70]]
                ]
            }
        }
        step_6 = self.color_latex(step_6, tex_to_color_map)
        
        storage_explanation = (
            Tex(
                r'Store the result as a new column of $Q$.',
                tex_environment='flushleft',
                font_size=34,
                color=BLACK
            ).next_to(step_6, DOWN, aligned_edge=LEFT, buff=0.6)
        )

        Q_tex = (
            MathTex(
                r'Q = \begin{bmatrix}  q^{(1)} & q^{(2)} \end{bmatrix}',
                color=BLACK,
                font_size=34
            ).next_to(storage_explanation, DOWN, buff=0.5)
        )
        tex_to_color_map = {
            r'\q_m^{*}': {
                'color': self.NORMED_RESIDUAL_COLOR,
                'indexes': [
                    [0,[3,4,5,6,7,8,9,10]]
                ]
            }
        }
        Q_tex = self.color_latex(Q_tex, tex_to_color_map)

        self.play(TransformMatchingShapes(storage_tex, Q_tex), FadeIn(step_6), FadeIn(storage_explanation))

        # ----- STEP 7: Solve the least squares problem
        self.next_slide()
        self.play(
            FadeOut(step_6),
            FadeOut(storage_explanation),
            Q_tex.animate.to_edge(LEFT).to_edge(UP)
        )

        step_7 = (
            Tex(
                r'\item \textbf{Step 7:} Find new boundary data $\psi_m^{(1)} $ \newline'
                r' by solving the least-squares problem',
                font_size=34,
                color=BLACK
            ).next_to(Q_tex, DOWN, aligned_edge=LEFT, buff=0.5)
        )
        tex_to_color_map = {
            r'\psi_m^{*}': {
                'color': self.PSI_COLOR,
                'indexes': [
                    [0,[25,26,27,28,29]]
                ]
            }
        }
        step_7 = self.color_latex(step_7, tex_to_color_map)

        lsq_problem = (
            MathTex(
                r'&\psi_m^{(1)} = \psi_m^{(0)} + y_1 q_m^{(1)} \\ '
                r'&y_1 = \operatorname{argmin} J(y) \\ '
                r'&J(y) = \lVert -\mathcal{B}(u_{inc}) - (\psi^{(0)} + y q^{(1)})\rVert',
                color=BLACK,
                font_size=34
            ).next_to(step_7, DOWN, aligned_edge=LEFT, buff=0.4)
        )
        tex_to_color_map = {
            r'\psi_m^{*}': {
                'color': self.PSI_COLOR,
                'indexes': [
                    [0,[0,1,2,3,4,6,7,8,9,10,48,49,50,51]]
                ]
            },
            r'\q_m^{*}': {
                'color': self.NORMED_RESIDUAL_COLOR,
                'indexes': [
                    [0,[14,15,16,17,18,54,55,56,57]]
                ]
            },
            r'u_{inc}': {
                'color': DARK_BROWN,
                'indexes': [
                    [0,[41,42,43,44]]
                ]
            }
        }
        lsq_problem = self.color_latex(lsq_problem, tex_to_color_map)

        error_explanation = (
            Tex(
                r'If $J(y_1) < \varepsilon$, terminate. Otherwise, repeat.',
                color=BLACK,
                font_size=34
            ).next_to(lsq_problem, DOWN, aligned_edge=LEFT, buff=1)
        )
        
        self.play(FadeIn(step_7), FadeIn(lsq_problem), FadeIn(error_explanation))

        # Step 8 - Repeat steps 4-7 until convergence 
        self.next_slide()
        self.play(
            FadeOut(error_explanation),
            FadeOut(lsq_problem),
            FadeOut(step_7),
            FadeOut(Q_tex)
        )

        step_8 = (
            Tex(
                r'\item \textbf{Step 8}: Repeat steps 4-7 until convergence:',
                color=BLACK,
                font_size=34
            ).to_edge(LEFT)
            .to_edge(UP)
        )

        precursor_step = (
            Tex(
                r'Given orthonormal set $Q = \begin{bmatrix}q^{(1)} & \ldots & q^{(n)} \end{bmatrix}$:',
                color=BLACK,
                font_size=29,
            ).next_to(step_8, DOWN, aligned_edge=LEFT, buff=0.7)
            .shift(RIGHT*0.1)
        )
        tex_to_color_map = {
            r'\q^{*}': {
                'color': self.NORMED_RESIDUAL_COLOR,
                'indexes': [
                    [0,[22,23,24,25,29,30,31,32]]
                ]
            }
        }
        precursor_step = self.color_latex(precursor_step, tex_to_color_map)

        steps = (
            Tex(
                r'\item Solve single-scattering BVPs with \newline boundary data $q_{m}^{(n)}$',
                r'\item Compute raw residuals \newline $\displaystyle r_m^{(n)} := \mathcal{B}_m \left(\sum_{\bar{m} \neq m} \mathcal{K}_{\bar{m},L}^{(n)} \right) + q_m^{(n)}$',
                r'\item Use Arnoldi Iteration to come up with \newline orthonormal vector $q^{(n+1)}$ to columns of $Q$',
                r'\item Solve least squares problem to recover \newline optimal boundary data $\psi_{m}^{(n+1)}$',
                color=BLACK,
                font_size=29,
                tex_environment='itemize'
            ).next_to(precursor_step, DOWN, aligned_edge=LEFT, buff=0.5)
        )
        tex_to_color_map = {
            r'\q^{*}': {
                'color': self.NORMED_RESIDUAL_COLOR,
                'indexes': [
                    [0,[43,44,45,46,47]],
                    [1,[48,49,50,51,52]],
                    [2,[49,50,51,52,53,54]]
                ]
            },
            r'\r^{*}': {
                'color': self.RESIDUAL_COLOR,
                'indexes': [
                    [1,[20,21,22,23,24]]
                ]
            },
            r'\mathcal{K}_{\bar{m},L}^{*}': {
                'color': PURE_RED,
                'indexes': [
                    [1,[37,38,39,40,41,42,43,44]]
                ]
            },
            r'\psi_{m}^{(n+1)}': {
                'color': self.PSI_COLOR,
                'indexes': [
                    [3,[53,54,55,56,57,58,59]]
                ]
            }
        }
        steps = self.color_latex(steps, tex_to_color_map)

        n_step_labels = VGroup()
        for m, obstacle in enumerate(obstacles):
            new_lbl = MathTex(
                rf'q_{m+1}^{{(n)}}',
                color=self.NORMED_RESIDUAL_COLOR,
                font_size=30
            ).next_to(obstacle, RIGHT*0.5)
            n_step_labels.add(new_lbl)    
        
        self.play(FadeIn(step_8), FadeIn(precursor_step), FadeIn(steps), FadeTransform(new_labels, n_step_labels))

        # Finally, we can reconstruct the solution based off of the final boundary data 
        self.next_slide()
        self.play(
            FadeOut(step_8),
            FadeOut(precursor_step),
            FadeOut(steps)
        )

        final_step_labels = VGroup()
        for m, obstacle in enumerate(obstacles):
            new_lbl = MathTex(
                rf'\psi_{m+1}^{{*}}',
                color=self.PSI_COLOR,
                font_size=30
            ).next_to(obstacle, RIGHT*0.5)
            final_step_labels.add(new_lbl)  

        final_step = (
            Tex(
                r'\item \textbf{Final Step:} Solve the BVP with optimal boundary data \newline $\psi_m^*$ found during iteration',
                color=BLACK,
                font_size=34
            ).to_edge(LEFT)
            .to_edge(UP)
        )
        tex_to_color_map = {
            r'\psi_{m}^{(n+1)}': {
                'color': self.PSI_COLOR,
                'indexes': [
                    [0,[44,45,46]]
                ]
            }
        }
        final_step = self.color_latex(final_step, tex_to_color_map)

        final_bvp = (
            MathTex(
                r'''
                & \Delta u_m^{*} + k^2 u_m^{*} = 0 \quad \text{ in } \Omega_m^{-} \\
                & \mathcal{B}_m u_m^{*} = \psi_m^{*} \quad \text{ on } \Gamma_m \\
                & \operatorname{MKFE}[u_m^{*}, \mathcal{K}_{m, L}^{*}] = 0 \quad \text{ on } \mathcal{C}_m
                ''',
                font_size=38,
                color=BLACK
            ).next_to(final_step, DOWN, aligned_edge=LEFT, buff=0.8)
        )
        tex_to_color_map = {
            r'u_m^{*}': {
                'color': PURE_BLUE,
                'indexes': [
                    [0,[1,2,3,7,8,9,19,20,21,35,36,37]]
                ]
            },
            r'\psi_m^{*}': {
                'color': self.PSI_COLOR,
                'indexes': [
                    [0,[23,24,25]]
                ]
            },
            r'\mathcal{K}_{m, L}^{*}': {
                'color': PURE_RED,
                'indexes': [
                    [0,[39,40,41,42,43]]
                ]
            }
        }
        final_bvp = self.color_latex(final_bvp, tex_to_color_map)


        self.play(
            FadeIn(final_step),
            FadeIn(final_bvp),
            FadeTransform(n_step_labels, final_step_labels),
            obstacles.animate.set(stroke_color=self.PSI_COLOR),
        )

        # Fade out everything left in the scene at this point 
        self.next_slide()
        self.play(
            *[FadeOut(mob)for mob in self.mobjects]
        )

        # Discuss implementation details
        title = (
            Tex(
                'Implementation Details',
                font_size=70,
                color=BLACK
            ).to_edge(UP)
        )

        details = (
            Tex(
                r'\item Impelemented using 2nd-order finite-difference scheme',
                r'\item The Karp Expansion terms $\mathcal{K}_{\bar{m},L}^{(n)}$ computed at other obstacle boundaries $\Gamma_m$ using 3rd-order Lagrange Interpolation',
                r'\item LU-decomposition of finite-difference matrices used',
                tex_environment='itemize',
                color=BLACK,
                font_size=35
            ).next_to(title, DOWN, buff=1)
        )
        tex_to_color_map = {
            r'\mathcal{K}_{\bar{m},L}': {
                'color': PURE_RED,
                'indexes': [
                    [1, [22,23,24,25,26,27,28,29]]
                ]
            }
        }
        details = self.color_latex(details, tex_to_color_map)
        
        self.play(FadeIn(title), FadeIn(details))

        self.next_slide()
        self.play(FadeOut(title), FadeOut(details))
    

    ## ................... RESULTS SLIDES .................... ##
    def results_2cyl_Horiz(self):
        PREFIX = '2cyl_Horiz'
        path_stem = f'figures/{PREFIX}/{PREFIX}_'
        total_wave_file = path_stem + 'total_wave.png'
        boundary_err_file = path_stem + 'bndry_max_error.png'
        obstacle_1_bndry = path_stem + 'obstacle_1_bndry.png'
        obstacle_2_bndry = path_stem + 'obstacle_2_bndry.png'


        title = (
            Tex(
                '2 Horizontal Cylinders',
                color=BLACK,
                font_size=55
            ).to_edge(UP).to_edge(LEFT)
        )
        
        total_wave = (
            ImageMobject(total_wave_file).shift(DOWN*0.4)
        )

        self.play(FadeIn(title), FadeIn(total_wave))

        # NEXT PART OF SLIDE 
        self.next_slide()
        self.play(FadeOut(total_wave))
        convergence_table = (
            Tex(
                r'''
                \begin{tabular}{c | c | c | c | c | c}
                    PPW & h & \#1 Error & \#1 Order & \#2 Error & \#2 Order \\
                    \hline
                    30  &  0.03333  & 8.04038e-03  &   &  8.85286e-03  & \\
                    \hline
                    40  & 0.02500  & 4.44243e-03  & 2.06226 & 4.85848e-03 & 2.08569 \\
                    \hline
                    50  &  0.02000  & 2.79463e-03 & 2.07715 & 3.06550e-03 & 2.06375 \\
                    \hline
                    60  &  0.01667  & 1.91671e-03  &  2.06826 & 2.10986e-03 & 2.04908
                \end{tabular}
                ''',
                color=BLACK,
                font_size=25,
            ).next_to(title,DOWN,aligned_edge=LEFT,buff=0.2)
        )
        boundary_max_err = (
            ImageMobject(boundary_err_file)
            .scale(0.8)
            .next_to(convergence_table, DOWN, aligned_edge=LEFT)
        )
        obs_1_bndry = (
            ImageMobject(obstacle_1_bndry)
            .scale(0.8)
            .next_to(boundary_max_err, RIGHT, aligned_edge=UP)
        )
        self.play(FadeIn(convergence_table), FadeIn(boundary_max_err), FadeIn(obs_1_bndry))
        self.next_slide()
        self.play(FadeOut(title), FadeOut(convergence_table), FadeOut(boundary_max_err), FadeOut(obs_1_bndry))


    def results_3cyl(self):
        PREFIX = '3cyl'
        path_stem = f'figures/{PREFIX}/{PREFIX}_'
        total_wave_file = path_stem + 'total_wave.png'
        boundary_err_file = path_stem + 'bndry_max_error.png'
        obstacle_1_bndry = path_stem + 'obstacle_1_bndry.png'
        obstacle_2_bndry = path_stem + 'obstacle_2_bndry.png'
        obstacle_3_bndry = path_stem + 'obstacle_3_bndry.png'

        

        title = (
            Tex(
                '3 Cylinders',
                color=BLACK,
                font_size=55
            ).to_edge(UP).to_edge(LEFT).shift(RIGHT*0.5)
        )
        
        total_wave = (
            ImageMobject(total_wave_file).shift(DOWN*0.4)
        )

        self.play(FadeIn(title), FadeIn(total_wave))

        # NEXT PART OF SLIDE 
        self.next_slide()
        self.play(FadeOut(total_wave))
        convergence_table = (
            Tex(
                r'''
                \begin{tabular}{c | c | c | c | c | c | c | c }
                   PPW  & h &  \#1 error & \#1 order  &  \#2 error  & \#2 order & \#3 error & \#3 order \\
                   \hline
                   30  & 0.03333  & 6.47411e-03 & &  7.29463e-03  & &    6.59540e-03  & \\ 
                   \hline
                   40  &  0.02500  & 3.56752e-03 &  2.07153 & 4.03376e-03  & 2.05936 & 3.64139e-03  & 2.06480 \\
                   \hline
                   50  &  0.02000 &  2.25059e-03  & 2.06450 & 2.55953e-03 & 2.03847 & 2.30684e-03 & 2.04570 \\
                   \hline
                   60 & 0.01667 & 1.55186e-03 & 2.03891 & 1.76414e-03 &  2.04125 &  1.59147e-03  & 2.03609 \\
                \end{tabular}
                ''',
                color=BLACK,
                font_size=25,
            ).next_to(title,DOWN,aligned_edge=LEFT,buff=0.2)
        )
        boundary_max_err = (
            ImageMobject(boundary_err_file)
            .scale(0.8)
            .next_to(convergence_table, DOWN, aligned_edge=LEFT)
            .shift(LEFT*0.8)
        )
        obs_2_bndry = (
            ImageMobject(obstacle_2_bndry)
            .scale(0.8)
            .next_to(boundary_max_err, RIGHT, aligned_edge=UP)
        )
        self.play(FadeIn(convergence_table), FadeIn(boundary_max_err), FadeIn(obs_2_bndry))
        self.next_slide()
        self.play(FadeOut(title), FadeOut(convergence_table), FadeOut(boundary_max_err), FadeOut(obs_2_bndry))


    def results_4cyl_modesto(self):
        PREFIX = '4cyl_Modesto'
        path_stem = f'figures/{PREFIX}/{PREFIX}_'
        total_wave_file = path_stem + 'total_wave.png'
        boundary_err_file = path_stem + 'bndry_max_error.png'
        obstacle_1_bndry = path_stem + 'obstacle_1_bndry.png'
        obstacle_2_bndry = path_stem + 'obstacle_2_bndry.png'
        obstacle_3_bndry = path_stem + 'obstacle_3_bndry.png'
        obstacle_4_bndry = path_stem + 'obstacle_3_bndry.png'

        

        title = (
            Tex(
                '4 Horizontal Cylinders',
                color=BLACK,
                font_size=55
            ).to_edge(UP).to_edge(LEFT).shift(RIGHT*0.5)
        )
        
        total_wave = (
            ImageMobject(total_wave_file).shift(DOWN*0.4)
        )

        self.play(FadeIn(title), FadeIn(total_wave))

        # NEXT PART OF SLIDE 
        self.next_slide()
        self.play(FadeOut(total_wave))
        convergence_table = (
            Tex(
                r'''
                \begin{tabular}{c | c | c | c | c | c | c | c | c | c}
                   PPW  & h &  \#1 error & \#1 order  &  \#2 error  & \#2 order & \#3 error & \#3 order & \#4 error & \#4 order  \\
                   \hline
                   30  & 0.03333  & 4.38240e-03 & &  4.59708e-03  & &  2.65432e-03   & &  1.91371e-03 & \\ 
                   \hline
                   40 & 0.02500 & 2.41845e-03 & 2.06642 & 2.56807e-03 &  2.02399 & 1.47907e-03 & 2.03270 & 1.06470e-03 &  2.03819 \\
                   \hline
                   50 &  0.02000 & 1.52994e-03 & 2.05203 & 1.64331e-03 & 2.00070 & 9.40470e-04 & 2.02914 & 6.76505e-04 & 2.03236 \\
                \end{tabular}
                ''',
                color=BLACK,
                font_size=25,
            ).next_to(title,DOWN,aligned_edge=LEFT,buff=0.2)
            .scale(0.9)
            .shift(LEFT*0.5)
        )
        boundary_max_err = (
            ImageMobject(boundary_err_file)
            .scale(0.8)
            .next_to(convergence_table, DOWN, aligned_edge=LEFT)
            .shift(LEFT*0.3)
        )
        obs_2_bndry = (
            ImageMobject(obstacle_2_bndry)
            .scale(0.8)
            .next_to(boundary_max_err, RIGHT, aligned_edge=UP)
        )
        self.play(FadeIn(convergence_table), FadeIn(boundary_max_err), FadeIn(obs_2_bndry))
        self.next_slide()
        self.play(FadeOut(title), FadeOut(convergence_table), FadeOut(boundary_max_err), FadeOut(obs_2_bndry))


    def results_7smile(self):
        PREFIX = '7smile'
        path_stem = f'figures/{PREFIX}/{PREFIX}_'
        total_wave_file = path_stem + 'total_wave.png'
        boundary_err_file = path_stem + 'bndry_max_error.png'
        farfield_pattern_file = path_stem + 'farfield_pattern.png'

        title = (
            Tex(
                '7 Cylinders in a Smile Shape',
                color=BLACK,
                font_size=55
            ).to_edge(UP).to_edge(LEFT).shift(RIGHT*0.5)
        )
        
        total_wave = (
            ImageMobject(total_wave_file).shift(DOWN*0.4)
        )

        self.play(FadeIn(title), FadeIn(total_wave))

        # NEXT PART OF SLIDE 
        self.next_slide()
        self.play(FadeOut(total_wave))
        convergence_table_1 = (
            Tex(
                r'''
                \begin{tabular}{c | c | c | c | c | c | c | c | c | c}
                   PPW  & h &  \#1 error & \#1 order  &  \#2 error  & \#2 order & \#3 error & \#3 order & \#4 error & \#4 order  \\
                   \hline
                   30  & 0.03333  & 1.26192e-02   &   &  8.80087e-03 & &   8.65337e-03 & &  1.11051e-02  & \\
                   \hline
                   40  & 0.02500  & 6.97659e-03  & 2.06013 & 4.88655e-03  & 2.04519 & 4.82399e-03  & 2.03123 & 6.17386e-03  & 2.04073 \\
                   \hline
                   50  & 0.02000 &  4.41514e-03  &  2.05034 & 3.10548e-03 & 2.03152 & 2.81281e-03 & 2.41735 & 3.91679e-03 & 2.03928 \\
                \end{tabular}
                ''',
                color=BLACK,
                font_size=25,
            ).next_to(title,DOWN,aligned_edge=LEFT,buff=0.2)
            .scale(0.9)
            .shift(LEFT*0.5)
        )
        convergence_table_2 = (
            Tex(
                r'''
                \begin{tabular}{c | c | c | c | c | c | c | c }
                   PPW  & h &  \#5 error & \#5 order  &  \#6 error  & \#6 order & \#7 error & \#7 order \\
                   \hline
                   30  & 0.03333  & 8.03806e-03 & &   9.49762e-03  & &  5.86872e-03  & \\
                   \hline
                   40  & 0.02500  & 4.49143e-03 &  2.02312 &  5.24066e-03 &  2.06684 &  3.23683e-03  &  2.06840 \\
                   \hline
                   50  & 0.02000 &  2.84306e-03 &  2.04931 & 3.31483e-03 &  2.05268 & 2.04712e-03  & 2.05320 \\
                \end{tabular}
                ''',
                color=BLACK,
                font_size=25,
            ).next_to(convergence_table_1,DOWN,aligned_edge=LEFT,buff=0.2)
            .scale(0.9)
            .shift(LEFT*0.5)
        )
        boundary_max_err = (
            ImageMobject(boundary_err_file)
            .scale(0.6)
            .to_edge(DOWN)
            .shift(DOWN*0.3)
        )
        
        self.play(FadeIn(convergence_table_1),FadeIn(convergence_table_2), FadeIn(boundary_max_err))

        
        self.next_slide()
        self.play(FadeOut(convergence_table_1),FadeOut(convergence_table_2), FadeOut(boundary_max_err))

        farfield_pattern = (
            ImageMobject(farfield_pattern_file).shift(DOWN*0.4)
        )
        self.play(FadeIn(farfield_pattern))

        self.next_slide()
        self.play(FadeOut(title), FadeOut(farfield_pattern))


    def results_slides(self):
        self.results_2cyl_Horiz()
        self.results_3cyl()
        self.results_4cyl_modesto()
        self.results_7smile()


    ## ................... FUTURE WORK SLIDES .................... ##
    def future_work_slides(self):
        title = (
            Tex(
                'Future Work',
                color=BLACK,
                font_size=65
            ).to_edge(UP)
        )

        results = (
            Tex(
                r'\item Experiment with larger arrays of obstacles with a variety of shapes',
                r'\item Introduce higher-order techniques in interior and examine convergence',
                r'\item Generalize to arbitrary physical boundary conditions',
                color=BLACK,
                font_size=35,
                tex_environment='itemize'
            ).next_to(title, DOWN, buff=1.0)
        )

        
        self.play(FadeIn(title), FadeIn(results))

        self.next_slide()
        self.play(FadeOut(title), FadeOut(results))


    def acknowledgements_slide(self):
        title = (
            Tex(
                'Acknowledgements',
                color=BLACK,
                font_size=65
            ).to_edge(UP)
        )

        acknowledgements = (
            Tex(
                r'\item Brigham Young University Math Department',
                r'\item Vianey Villamizar, Brigham Young University',
                r'\item Jesse Wayment, Brigham Young University',
                r'\item Manim Community Edition',
                color=BLACK,
                font_size=35,
                tex_environment='itemize'
            ).next_to(title, DOWN, buff=1.0)
        )

        self.play(FadeIn(title), FadeIn(acknowledgements))

    
    ##########################################################
    ## ................... MAIN METHOD .................... ##
    ##########################################################
    def construct(self):
        # Setup constants and templates 
        self.add_to_tex_template()

        self.PSI_COLOR = ManimColor.from_hex('#017d1e')
        self.RESIDUAL_COLOR = ManimColor.from_hex('#590561')
        self.NORMED_RESIDUAL_COLOR = self.RESIDUAL_COLOR

        # ------------------------ INTRO ------------------------- #
        # TITLE SLIDE
        title_slide_objs = self.title_slide()
        self.play(FadeIn(title_slide_objs))
        self.next_slide()
        
        # ROADMAP SLIDE
        self.play(FadeOut(title_slide_objs)) 
        roadmap_slide_objs = self.roadmap_slide()
        self.play(FadeIn(roadmap_slide_objs))
        self.next_slide()

        # ------------------------ OVERVIEW ---------------------- #
        # SECTION INTRO SLIDE         
        self.play(roadmap_slide_objs['roadmap_sections'][1:].animate.set_opacity(0.2))
        self.next_slide()
        self.play(FadeOut(roadmap_slide_objs))

        # SCATTERING ANIMATION/MOTIVATION
        scene = self.overview_slides()
        
        self.motivating_equation_slides(scene)

        # --------------- BOUNDARY CONDITION FORMULATION ---------------- #
        roadmap_slide_objs = self.roadmap_slide()
        roadmap_slide_objs['roadmap_sections'][0].set_opacity(0.2)
        roadmap_slide_objs['roadmap_sections'][1].set_opacity(1)
        roadmap_slide_objs['roadmap_sections'][2:].set_opacity(0.2)
        self.play(FadeIn(roadmap_slide_objs))
        self.next_slide()
        self.play(FadeOut(roadmap_slide_objs))

        scattering_scene = self.get_scattering_scene()
        self.karp_slides(scattering_scene)

        # --------------- RESIDUAL METHOD FORMULATION ---------------- #
        # Display roadmap slide for next section
        roadmap_slide_objs = self.roadmap_slide()
        roadmap_slide_objs['roadmap_sections'][0:2].set_opacity(0.2)
        roadmap_slide_objs['roadmap_sections'][2].set_opacity(1)
        roadmap_slide_objs['roadmap_sections'][3:].set_opacity(0.2)
        self.play(FadeIn(roadmap_slide_objs))

        self.next_slide()
        self.play(FadeOut(roadmap_slide_objs))
        
        slide_objs = self.boundary_guess_slides()

        self.residual_method_slides(slide_objs)
        
        # --------------- RESULTS SECTION ---------------- #
        # Display roadmap slide for next section 
        roadmap_slide_objs = self.roadmap_slide()
        roadmap_slide_objs['roadmap_sections'][0:3].set_opacity(0.2)
        roadmap_slide_objs['roadmap_sections'][3].set_opacity(1)
        roadmap_slide_objs['roadmap_sections'][4:].set_opacity(0.2)
        self.play(FadeIn(roadmap_slide_objs))
        
        self.next_slide()
        self.play(FadeOut(roadmap_slide_objs))

        self.results_slides()

        # ------------ FUTURE WORK SECTION -------------- #
        roadmap_slide_objs = self.roadmap_slide()
        roadmap_slide_objs['roadmap_sections'][0:4].set_opacity(0.2)
        roadmap_slide_objs['roadmap_sections'][4].set_opacity(1)
        self.play(FadeIn(roadmap_slide_objs))
        
        self.next_slide()
        self.play(FadeOut(roadmap_slide_objs))

        self.future_work_slides()

        self.acknowledgements_slide()