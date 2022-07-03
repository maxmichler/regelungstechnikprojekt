from this import d
import numpy as np
import scipy.linalg as sla
import matplotlib.pyplot as plt
import sympy

from libmimo import mimo_rnf, mimo_acker, poly_transition
from scipy.linalg import solve_continuous_are
from scipy.linalg import solve_discrete_are
from scipy.signal import cont2discrete
from scipy.integrate import quad
from sympy import *
from sympy.abc import x


class LinearizedSystem:
    def __init__(self, A, B, C, D, x_equi, u_equi, y_equi):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.x_equi = x_equi
        self.u_equi = u_equi
        self.y_equi = y_equi

    # Ackermannformel
    def acker(self, eigs):
        return mimo_acker(self.A, self.B, eigs)

    # Berechnung des Ausgangs
    def output(self, t, x, controller):
        # Regler auswerten (wichtig falls Durchgriff existiert)
        u = controller(t, x)
        if x.ndim == 1:
            y = self.C@(x-self.x_equi) + self.y_equi + self.D@(u-self.u_equi)
        else:
            x_equi = self.x_equi.reshape((self.x_equi.shape[0], 1))
            u_equi = self.u_equi.reshape((self.u_equi.shape[0], 1))
            y_equi = self.y_equi.reshape((self.y_equi.shape[0], 1))
            y = self.C@(x-x_equi)+y_equi+self.D@(u-u_equi)
        return y


class DiscreteLinearizedSystem(LinearizedSystem):
    def __init__(self, A, B, C, D, x_equi, u_equi, y_equi, Ta):
        super().__init__(A, B, C, D, x_equi, u_equi, y_equi)
        self.Ta = 0

    # Quadratisch optimaler Regler
    def lqr(self, Q, R, S):
        ######-------!!!!!!Aufgabe!!!!!!-------------########
        # Hier sollten die korrekten Reglerverstärkungen berechnet werden
        K_lqr = np.zeros((R.shape[0], Q.shape[0]))
        return K_lqr
        ######-------!!!!!!Aufgabe Ende!!!!!!-------########

    def rest_to_rest_trajectory(self, ya, yb, N, kronecker, maxderi=None):
        return DiscreteFlatnessBasedTrajectory(self, ya, yb, N, kronecker, maxderi)


class ContinuousLinearizedSystem(LinearizedSystem):
    def __init__(self, A, B, C, D, x_equi, u_equi, y_equi):
        super().__init__(A, B, C, D, x_equi, u_equi, y_equi)

    def discretize(self, Ta):
        assert Ta > 0
        ######-------!!!!!!Aufgabe!!!!!!-------------########
        A_d = sla.expm(self.A*Ta)
        B_d = Matrix(self.A*x).exp().integrate((x, 0, Ta)) @ self.B
        B_d = np.array(B_d).astype(np.cfloat)
        C_d = self.C
        D_d = self.D
        return DiscreteLinearizedSystem(A_d, B_d, C_d, D_d, self.x_equi, self.u_equi, self.y_equi, Ta)
        ######-------!!!!!!Aufgabe Ende!!!!!!-------########

    # Quadratisch optimaler Regler
    def lqr(self, Q, R, S):
        ######-------!!!!!!Aufgabe!!!!!!-------------########
        # Hier sollten die korrekten Reglerverstärkungen berechnet werden
        K_lqr = np.zeros((R.shape[0], Q.shape[0]))
        return K_lqr
        ######-------!!!!!!Aufgabe Ende!!!!!!-------########

    def rest_to_rest_trajectory(self, ya, yb, T, kronecker, maxderi=None):
        return ContinuousFlatnessBasedTrajectory(self, ya, yb, T, kronecker, maxderi)

    # linearisiertes Zustandsraumodell des Systems
    def model(self, t, x, controller):
        # t: Zeit
        # Zustand
        # x[0]: Elevations-Winkel epsilon
        # x[1]: Azimutwinkelwinkelgeschwindigkeit \dot{\alpha}
        # x[2]: Winkelgeschwindigkeit \dot{\epsilon}
        # controller(t,x): Solldrehzahlen werden als Funktion übergeben

        dim_u = self.u_equi.shape[0]
        dim_x = self.x_equi.shape[0]
        dim_y = self.y_equi.shape[0]

        # Auslenkung aus Ruhelage
        x_rel = np.array(x).flatten()-self.x_equi

        # Eingang auswerten
        u = controller(t, x)
        u = np.array(u).flatten()

        # Auslenkung aus Ruhelage
        u_rel = u-self.u_equi
        x_rel = x_rel.flatten()

        # Zustand auspacken
        dx = (self.A@x_rel+self.B@u_rel).flatten()

        # Ableitungen zurückgeben
        return dx


class Heli:
    """Modell des Heliracks"""

    def __init__(self):
        self.Vmax = 0.024
        self.J1 = 0.002
        self.J2 = 0.012
        self.J3 = 0.013
        self.J4 = 1e-05
        self.J5 = 1e-05
        self.c1 = 0.001
        self.c2 = 0.001
        self.s1 = 8e-7
        self.s2 = 8e-7
        self.d1 = 0.272
        self.d2 = 0.272
        self.Ta = 0.1

    def equilibrium(self, y):

        # Elevationwinkel
        epsilon = y[0]

        # Rotationsgeschwindigkeit des Auslegers um die  Vertikale
        dot_alpha = y[1]

        ######-------!!!!!!Aufgabe!!!!!!-------------########
        # Hier die sollten die korrekten Ruhelagen in Abhängigkeit des zugehörigen Ausgangs berechnet werden

        # Winkelfunktionen vorausberechnen
        ceps = np.cos(epsilon)
        seps = np.sin(epsilon)
        s2eps = np.sin(2*epsilon)

        F_alpha = self.c1*dot_alpha/(self.d1*ceps)
        if(F_alpha >= 0):
            w_alpha = np.sqrt(self.c1*dot_alpha/(self.d2*ceps*self.s1))
        else:
            w_alpha = np.sqrt(-self.c1*dot_alpha/(self.d2*ceps*self.s1))

        p = -self.J4*seps*dot_alpha/(self.d2*self.s2)
        q = (-self.Vmax*ceps-1/2*(self.J2+self.J4)
             * s2eps*dot_alpha**2)/(self.d2*self.s2)

        if((p/2)**2-q < 0):  # w_epsilon kann nur Realteil besitzen - Fallunterscheidung, da abs(x)*x nicht dasselbe wie x^2
            p = -1*p
            q = -1*q

        w_epsilon_1 = -p/2+np.sqrt((p/2)**2-q)
        w_epsilon_2 = -p/2-np.sqrt((p/2)**2-q)

        # Fallunterscheidung: w_epsilon in Gleichung einsetzten, tatsächliche Lösung finden, aufgrund numerischen Ungenauigkeiten Bereich um 0
        test_1 = abs(w_epsilon_1)*w_epsilon_1+p*w_epsilon_1+q
        test_2 = abs(w_epsilon_2)*w_epsilon_2+p*w_epsilon_2+q

        if(test_1 > -0.0001 and test_1 < 0.0001):
            w_epsilon = w_epsilon_1
        elif(test_2 > -0.0001 and test_2 < 0.0001):
            w_epsilon = w_epsilon_2
        else:
            w_epsilon = 0

        p_alpha = (self.J1+(self.J2+self.J4)*ceps**2)*dot_alpha + \
            self.J4*ceps*w_epsilon  # Gleichung 1a
        p_epsilon = self.J5*w_alpha  # Gleichung 1b

        x = np.array([epsilon, p_alpha, p_epsilon])
        u = np.array([w_alpha, w_epsilon])
        ######-------!!!!!!Aufgabe Ende!!!!!!-------########

        return x, u

    def linearize(self, y_equi, debug=False):
        # Berechnung der Systemmatrizen des linearen Systems

        # Ruhelage berechnen
        x_equi, u_equi = self.equilibrium(y_equi)

        ######-------!!!!!!Aufgabe!!!!!!-------------########
        epsilon, p_epsilon, p_alpha, w_alpha, w_abs_alpha, w_epsilon, w_abs_epsilon = symbols(
            'epsilon p_epsilon p_alpha w_alpha w_abs_alpha w_epsilon w_abs_epsilon')

        dot_alpha = (p_alpha - self.J4*cos(epsilon)*w_epsilon) / \
            (self.J1+(self.J2+self.J4)*cos(epsilon)**2)

        G = Matrix([[(p_epsilon-self.J5*w_alpha)/(self.J3+self.J5)],
                    [self.d1*cos(epsilon)*self.s1*w_abs_alpha *
                     w_alpha-self.c1*dot_alpha],
                    [-self.Vmax*cos(epsilon)-1/2*(self.J2+self.J4)*sin(2*epsilon)*dot_alpha**2-self.J4*w_epsilon*sin(epsilon)*dot_alpha+self.d2*self.s2*w_abs_epsilon*w_epsilon-self.c2*(p_epsilon-self.J5*w_alpha)/(self.J3+self.J5)]])

        A = diff(G, epsilon)
        A = A.col_insert(1, diff(G, p_alpha))
        A = A.col_insert(2, diff(G, p_epsilon))

        A = A.subs({epsilon: x_equi[0], p_alpha: x_equi[1], p_epsilon: x_equi[2], w_alpha: u_equi[0], w_abs_alpha: abs(
            u_equi[0]), w_epsilon: u_equi[1], w_abs_epsilon: abs(u_equi[1])})
        A = np.array(A).astype(np.float64)

        B = diff(G, w_alpha)+diff(G, w_abs_alpha)
        B = B.col_insert(1, diff(G, w_epsilon)+diff(G, w_abs_epsilon))

        B = B.subs({epsilon: x_equi[0], p_alpha: x_equi[1], p_epsilon: x_equi[2], w_alpha: u_equi[0], w_abs_alpha: abs(
            u_equi[0]), w_epsilon: u_equi[1], w_abs_epsilon: abs(u_equi[1])})
        B = np.array(B).astype(np.float64)

        y = Matrix([[epsilon], [dot_alpha]])  # Ausgangsvektor

        C = diff(y, epsilon)
        C = C.col_insert(1, diff(y, p_alpha))
        C = C.col_insert(2, diff(y, p_epsilon))

        C = C.subs({epsilon: x_equi[0], p_alpha: x_equi[1], p_epsilon: x_equi[2], w_alpha: u_equi[0], w_abs_alpha: abs(
            u_equi[0]), w_epsilon: u_equi[1], w_abs_epsilon: abs(u_equi[1])})
        C = np.array(C).astype(np.float64)

        D = diff(y, w_alpha)+diff(y, w_abs_alpha)
        D = D.col_insert(1, diff(y, w_epsilon)+diff(y, w_abs_epsilon))

        D = D.subs({epsilon: x_equi[0], p_alpha: x_equi[1], p_epsilon: x_equi[2], w_alpha: u_equi[0], w_abs_alpha: abs(
            u_equi[0]), w_epsilon: u_equi[1], w_abs_epsilon: abs(u_equi[1])})
        D = np.array(D).astype(np.float64)
        ######-------!!!!!!Aufgabe Ende!!!!!!-------########

        return ContinuousLinearizedSystem(A, B, C, D, x_equi, u_equi, y_equi)

    def verify_linearization(self, linear_model, eps_x=1e-6, eps_u=1e-6):
        """Compare linear state space model with its approximate Taylor linearization of non-linear model
        Parameters
        ----------
        linear_model : object of type LinearizedModel

        Returns
        -------
        nothing

        Notes
        -----
        Tay

        """
        A_approx = np.zeros_like(linear_model.A)
        B_approx = np.zeros_like(linear_model.B)
        C_approx = np.zeros_like(linear_model.C)
        D_approx = np.zeros_like(linear_model.D)
        x_equi = linear_model.x_equi
        u_equi = linear_model.u_equi
        dim_x = 3
        dim_u = 2
        dim_y = 2

        if np.isscalar(eps_x):
            eps_x = np.ones((dim_x,))*eps_x
        if np.isscalar(eps_u):
            eps_u = np.ones((dim_u,))*eps_u

        for jj in range(dim_x):
            def _fnu(t, x): return u_equi
            x_equi1 = np.array(x_equi)
            x_equi2 = np.array(x_equi)
            x_equi2[jj] += eps_x[jj]
            x_equi1[jj] -= eps_x[jj]
            # print("Ruhelage:",x_equi)
            dx = (self.model(0, x_equi2, _fnu) -
                  self.model(0, x_equi1, _fnu))/2/eps_x[jj]
            A_approx[:, jj] = dx
            dy = (self.output(0, x_equi2, _fnu) -
                  self.output(0, x_equi1, _fnu))/2/eps_x[jj]
            C_approx[:, jj] = dy

        error_A = np.abs(linear_model.A-A_approx)
        idx = np.unravel_index(np.argmax(error_A, axis=None), error_A.shape)
        print("Maximaler absoluter Fehler in Matrix A Zeile " +
              str(idx[0]+1)+", Spalte "+str(idx[1]+1)+" beträgt", str(error_A[idx[0], idx[1]])+".")
        scale_A = np.hstack([np.where(abs(A_approx[:, jj:jj+1]) > eps_x[jj],
                            abs(A_approx[:, jj:jj+1]), eps_x[jj]) for jj in range(dim_x)])
        error_rel_A = error_A/scale_A
        idx = np.unravel_index(
            np.argmax(error_rel_A, axis=None), error_rel_A.shape)
        print("Maximaler relativer Fehler in Matrix A Zeile " +
              str(idx[0]+1)+", Spalte "+str(idx[1]+1)+" beträgt", str(error_rel_A[idx[0], idx[1]])+".")

        error_C = np.abs(linear_model.C-C_approx)
        idx = np.unravel_index(np.argmax(error_C, axis=None), error_C.shape)
        print("Maximaler absoluter Fehler in Matrix C Zeile " +
              str(idx[0]+1)+", Spalte "+str(idx[1]+1)+" beträgt", str(error_C[idx[0], idx[1]])+".")
        scale_C = np.hstack([np.where(abs(C_approx[:, jj:jj+1]) > eps_x[jj],
                            abs(C_approx[:, jj:jj+1]), eps_x[jj]) for jj in range(dim_x)])
        error_rel_C = error_C/scale_C
        idx = np.unravel_index(
            np.argmax(error_rel_C, axis=None), error_rel_C.shape)
        print("Maximaler relativer Fehler in Matrix C Zeile " +
              str(idx[0]+1)+", Spalte "+str(idx[1]+1)+" beträgt", str(error_rel_C[idx[0], idx[1]])+".")

        for jj in range(dim_u):
            url1 = np.array(u_equi)
            url2 = np.array(u_equi)
            eps_u[jj] = 1e-6
            url1[jj] -= eps_u[jj]
            url2[jj] += eps_u[jj]
            def _fnu1(t, x): return url1
            def _fnu2(t, x): return url2
            dx = (self.model(0, x_equi, _fnu2) -
                  self.model(0, x_equi, _fnu1))/2/eps_u[jj]
            B_approx[:, jj] = dx
            dy = (self.output(0, x_equi, _fnu2) -
                  self.output(0, x_equi, _fnu1))/2/eps_x[jj]
            D_approx[:, jj] = dy
        error_B = np.abs(linear_model.B-B_approx)
        idx = np.unravel_index(np.argmax(error_B, axis=None), error_B.shape)
        print("Maximaler absoluter Fehler in Matrix B Zeile " +
              str(idx[0]+1)+", Spalte "+str(idx[1]+1)+" beträgt", str(error_B[idx[0], idx[1]])+".")
        scale_B = np.hstack([np.where(abs(B_approx[:, jj:jj+1]) > eps_x[jj],
                            abs(B_approx[:, jj:jj+1]), eps_x[jj]) for jj in range(dim_u)])
        error_rel_B = error_B/scale_B
        idx = np.unravel_index(
            np.argmax(error_rel_B, axis=None), error_rel_B.shape)
        print("Maximaler relativer Fehler in Matrix B Zeile " +
              str(idx[0]+1)+", Spalte "+str(idx[1]+1)+" beträgt", str(error_rel_B[idx[0], idx[1]])+".")

        error_D = np.abs(linear_model.D-D_approx)
        idx = np.unravel_index(np.argmax(error_D, axis=None), error_D.shape)
        print("Maximaler absoluter Fehler in Matrix D Zeile " +
              str(idx[0]+1)+", Spalte "+str(idx[1]+1)+" beträgt", str(error_D[idx[0], idx[1]])+".")
        scale_D = np.hstack([np.where(abs(D_approx[:, jj:jj+1]) > eps_x[jj],
                            abs(D_approx[:, jj:jj+1]), eps_x[jj]) for jj in range(dim_u)])
        error_rel_D = error_D/scale_D
        idx = np.unravel_index(
            np.argmax(error_rel_D, axis=None), error_rel_D.shape)
        print("Maximaler relativer Fehler in Matrix D Zeile " +
              str(idx[0]+1)+", Spalte "+str(idx[1]+1)+" beträgt", str(error_rel_D[idx[0], idx[1]])+".")
        return A_approx, B_approx, C_approx

    # Umrechnung zwischen Winkelgeschwindigkeiten und Impulsen
    def velocities_to_momentums(self, q, dq, u):
        p = np.zeros_like(dq)
        bJ2 = self.J2+self.J4
        bJ3 = self.J3+self.J5
        if dq.ndim == 1:
            p[0] = self.J5*u[1]+bJ3*dq[0]
            p[1] = self.J4*np.cos(q[0])*u[0] + \
                (self.J1+bJ2*np.cos(q[0])**2)*dq[1]
        else:
            p[0, :] = self.J5*u[1, :]+bJ3*dq[0, :]
            p[1, :] = self.J4*np.cos(q[0, :])*u[0, :] + \
                (self.J1+bJ2*np.cos(q[0, :])**2)*dq[1, :]
        return p

    # Nichtlineares Zustandsraumodell des Heli-Racks
    def model(self, t, x, controller):
        # t: Zeit
        # Zustand
        # x[0]: Elevations-Winkel epsilon
        # x[1]: Drehimpuls p_alpha
        # x[2]: Drehimpuls p_epsilon
        # controller(t,x): Solldrehzalen werden als Funktion übergeben

        # Eingang auswerten
        u = controller(t, x).flatten()
        assert np.shape(u) == (2,)

        # Zustand auspacken
        epsilon = x[0]
        p_alpha = x[1]
        p_epsilon = x[2]

        # Winkelfunktionen vorausberechnen
        ceps = np.cos(epsilon)
        seps = np.sin(epsilon)

        ######-------!!!!!!Aufgabe!!!!!!-------------########
        # Hier sollten die korrekten Ableitungen berechnet und zurückgegebenn werden
        #u[1] = w_e, u[0]=w_a

        dot_alpha = (p_alpha-self.J4*ceps*u[1]) / \
            (self.J1+(self.J2+self.J4)*ceps **
             2)  # Gleichung 1a auf dot_alpha umgeformt
        dot_epsilon = (p_epsilon-self.J5*u[0]) / \
            (self.J3+self.J5)  # Gleichung 1b

        F_alpha = self.s1*abs(u[0])*u[0]  # Gleichung 3
        F_eps = self.s2*abs(u[1])*u[1]  # Gleichung 3

        D_eps = self.c2*dot_epsilon  # Gleichung 4
        D_alpha = self.c1*dot_alpha  # Gleichung 4

        dot_p_alpha = self.d1*ceps*F_alpha-D_alpha  # Gleichung 2a
        dot_p_epsilon = -self.Vmax*ceps-1/2*(self.J2+self.J4)*np.sin(
            2*epsilon)*dot_alpha**2-self.J4*u[1]*seps*dot_alpha+self.d1*F_eps-D_eps  # Gleichung 2b

        dx = np.array([dot_epsilon, dot_p_alpha, dot_p_epsilon])

        ######-------!!!!!!Aufgabe Ende!!!!!!-------########
        return dx

    def output(self, t, x, controller):
        # Berechnung des Systemausgangs
        # Eingang auswerten
        u = controller(t, x)
        ######-------!!!!!!Aufgabe!!!!!!-------------########
        # Hier sollten die korrekten Ausgänge berechnet werden

        # Zustand auspacken
        epsilon = x[0]
        p_alpha = x[1]
        p_epsilon = x[2]

        # Winkelfunktionen vorausberechnen
        ceps = np.cos(epsilon)
        seps = np.sin(epsilon)

        dot_alpha = (p_alpha-self.J4*ceps*u[1]) / \
            (self.J1+(self.J2+self.J4)*ceps **
             2)  # Gleichung 1a auf dot_alpha umgeformt

        y = np.array([epsilon, dot_alpha])
        ######-------!!!!!!Aufgabe Ende!!!!!!-------########
        return y

    def generate_model_test_data(self, filename, count):
        import pickle
        x = np.random.rand(3, count)
        u = np.random.rand(2, count)
        dx = np.zeros_like(x)
        y = np.zeros_like(u)
        for ii in range(count):
            dx[:, ii] = self.model(0, x[:, ii], lambda t, x: u[:, ii])
            y[:, ii] = self.output(0, x[:, ii], lambda t, x: u[:, ii])

        with open(filename, 'wb') as f:
            pickle.dump([x, u, dx, y], f)
        f.close()

    def verify_model(self, filename):
        import pickle
        with open(filename, 'rb') as f:
            x, u, dx_load, y_load = pickle.load(f)
        f.close()
        dx = np.zeros_like(x)
        y = np.zeros_like(u)
        count = dx.shape[1]
        for ii in range(count):
            dx[:, ii] = self.model(0, x[:, ii], lambda t, x: u[:, ii])
            y[:, ii] = self.output(0, x[:, ii], lambda t, x: u[:, ii])
        error_dx_abs_max = np.max(np.linalg.norm(dx-dx_load, 2, axis=0))
        error_dx_rel_max = np.max(np.linalg.norm(
            dx-dx_load, 2, axis=0)/np.linalg.norm(dx_load, 2, axis=0))
        error_y_abs_max = np.max(np.linalg.norm(y-y_load, 2, axis=0))
        error_y_rel_max = np.max(np.linalg.norm(
            y-y_load, 2, axis=0)/np.linalg.norm(y_load, 2, axis=0))
        dx_load_max = np.max(np.linalg.norm(dx_load, 2, axis=0))
        y_load_max = np.max(np.linalg.norm(y_load, 2, axis=0))
        print("Maximaler absoluter Fehler in Modellgleichung (euklidische Norm):",
              error_dx_abs_max)
        print("Maximaler relativer Fehler in Modellgleichung (euklidische Norm):",
              error_dx_rel_max)
        print("Maximaler absoluter Fehler in Ausgangsgleichung (euklidische Norm):", error_y_abs_max)
        print("Maximaler relativer Fehler in Ausgangsgleichung (euklidische Norm):", error_y_rel_max)


class ContinuousFlatnessBasedTrajectory:
    """Zeitkontinuierliche flachheitsbasierte Trajektorien-Planung zum Arbeitspunktwechsel 
    für das lineare zeitkontinuierliche Modelle, die aus der Linearisierung im Arbeitspunkt abgeleitete worden sind.

    Args:
       ya, ye (numpy.array):  Anfangs- und Endwerte den Ausgang (absolut)
       T (float): Überführungszeit
       linearized_system: Entwurfsmodel
       kronecker: zu verwendende Steuerbarkeitsindizes
       maxderi: maximal Differenzierbarkeitsanforderungen für flachen Ausgang (None entspricht maxderi=kronecker)
    """

    def __init__(self, linearized_system, ya, yb, T, kronecker, maxderi=None):
        self.linearized_system = linearized_system
        self.T = T
        ya_rel = np.array(ya)-linearized_system.y_equi
        yb_rel = np.array(yb)-linearized_system.y_equi
        self.kronecker = np.array(kronecker, dtype=int)
        if maxderi == None:
            self.maxderi = self.kronecker
        else:
            self.maxderi = self.maxderi

        ######-------!!!!!!Aufgabe!!!!!!-------------########
        # Q Transformationsmatrix
        # S Kalmannsche Steuerbarkeitsmatrix
        self.A_rnf, Brnf, Crnf, self.M, self.Q, S = mimo_rnf(
            linearized_system.A, linearized_system.B, linearized_system.C, kronecker)
        ######-------!!!!!!Aufgabe Ende!!!!!!-------########

        # Umrechnung stationäre Werte zwischen Ausgang und flachem Ausgang
        ######-------!!!!!!Aufgabe!!!!!!-------------########
        # Achtung: Hier sollten alle werte relativ zum Arbeitspunkt angegeben werden
        # y = C * x + D * u
        # x_2 oder x_3 = 0, da dot{eta_1} oder dot{eta_2} = 0 sind - stationärer Fall
        xa, ua_equi = Heli().equilibrium(ya)
        xb, ub_equi = Heli().equilibrium(yb)
        if kronecker == (1, 2):
            self.eta_a = np.linalg.inv(Crnf[:, 0:2]) @ (ya_rel-np.dot(
                self.linearized_system.D, ua_equi - self.linearized_system.u_equi))
            self.eta_b = np.linalg.inv(Crnf[:, 0:2]) @ (yb_rel-np.dot(
                self.linearized_system.D, ub_equi - self.linearized_system.u_equi))
        elif kronecker == (2, 1):
            C = np.array([[Crnf[0, 0], Crnf[0, 2]], [Crnf[1, 0], Crnf[1, 2]]])
            self.eta_a = np.linalg.inv(
                C) @ (ya_rel-np.dot(self.linearized_system.D, ua_equi - self.linearized_system.u_equi))
            self.eta_b = np.linalg.inv(
                C) @ (yb_rel-np.dot(self.linearized_system.D, ub_equi - self.linearized_system.u_equi))
        ######-------!!!!!!Aufgabe Ende!!!!!!-------########

    # Trajektorie des flachen Ausgangs

    def flat_output(self, t, index, derivative):
        tau = t / self.T
        if derivative == 0:
            return self.eta_a[index] + (self.eta_b[index] - self.eta_a[index]) * poly_transition(tau, 0, self.maxderi[index])
        else:
            return (self.eta_b[index] - self.eta_a[index]) * poly_transition(tau, derivative, self.maxderi[index])/self.T**derivative

    # Zustandstrajektorie
    def state(self, t):
        tv = np.atleast_1d(t)
        dim_u = np.size(self.linearized_system.u_equi)
        dim_x = np.size(self.linearized_system.x_equi)
        dim_t = np.size(tv)
        ######-------!!!!!!Aufgabe!!!!!!-------------########
        # Berechnung eta mit Funktion flat_output und anschließende Rücktransformation aus RNG mit x = Q^-1*xrnf
        eta = list()
        for index in range(dim_u):
            eta = eta+[self.flat_output(tv, index, deri)
                       for deri in range(self.kronecker[index])]
        xrnf = np.vstack(eta)

        x = np.linalg.inv(self.Q) @ xrnf
        # Aufgrund von Broadcast-Regeln in Python mit den Shapes nur so berechnenbar
        state = np.transpose(x.transpose() + self.linearized_system.x_equi)

        if (np.isscalar(t)):
            state = state[:, 0]
        ######-------!!!!!!Aufgabe Ende!!!!!!-------########
        return state

    # Ausgangstrajektorie
    def output(self, t):
        tv = np.atleast_1d(t)
        dim_u = np.size(self.linearized_system.u_equi)
        dim_y = np.size(self.linearized_system.y_equi)
        dim_x = np.size(self.linearized_system.x_equi)

        x_abs = self.state(tv)
        u_abs = self.input(tv)
        x_rel = x_abs-self.linearized_system.x_equi.reshape((dim_x, 1))
        u_rel = u_abs-self.linearized_system.u_equi.reshape((dim_u, 1))
        y_rel = self.linearized_system.C@x_rel+self.linearized_system.D@u_rel
        y_abs = y_rel+self.linearized_system.y_equi.reshape((dim_y, 1))
        if (np.isscalar(t)):
            y_abs = result[:, 0]
        return y_abs

    # Eingangstrajektorie
    def input(self, t):
        tv = np.atleast_1d(t)
        dim_u = np.size(self.linearized_system.u_equi)
        eta = list()
        for index in range(dim_u):
            eta = eta+[self.flat_output(tv, index, deri)
                       for deri in range(self.kronecker[index])]
        xrnf = np.vstack(eta)
        v = -self.A_rnf[self.kronecker.cumsum()-1, :]@xrnf
        for jj in range(self.kronecker.shape[0]):
            v[jj, :] += self.flat_output(tv, jj, self.kronecker[jj])
        result = (np.linalg.inv(self.M)@v) + \
            self.linearized_system.u_equi.reshape((dim_u, 1))
        if (np.isscalar(t)):
            result = result[:, 0]
        return result


class DiscreteFlatnessBasedTrajectory:
    """Zeitdiskrete flachheitsbasierte Trajektorien-Planung zum Arbeitspunktwechsel 
    für lineare Modelle, die aus der Linearisierung im Arbeitspunkt abgeleitet worden sind.

    Args:
       ya, ye (numpy.array):  Anfangs- und Endwerte den Ausgang (absolut)
       T (float): Überführungszeit
       linearized_system: zeitdiskretes Entwurfsmodel
       kronecker: zu verwendende Steuerbarkeitsindizes
       maxderi: maximale Differenzierbarkeitsanforderungen für die Komponenten des flachen Ausgangs (bei None werden die Kronecker-Indizes gewählt; maxderi=kronecker)
    """

    def __init__(self, linearized_system, ya, yb, N, kronecker, maxderi=None):
        self.linearized_system = linearized_system
        self.N = N

        dim_u = np.size(linearized_system.u_equi)
        dim_x = np.size(linearized_system.x_equi)

        # Abstand von der Ruhelage berechnen
        ya_rel = np.array(ya)-linearized_system.y_equi
        yb_rel = np.array(yb)-linearized_system.y_equi
        self.kronecker = np.array(kronecker, dtype=int)

        # Glattheitsanforderungen an Trajektorie
        if maxderi == None:
            self.maxderi = self.kronecker
        else:
            self.maxderi = self.maxderi

        # Matrizen der Regelungsnormalform holen
        ######-------!!!!!!Aufgabe!!!!!!-------------########
        self.A_rnf, Brnf, Crnf, self.M, self.Q, S = mimo_rnf(
            linearized_system.A, linearized_system.B, linearized_system.C, kronecker)
        ######-------!!!!!!Aufgabe Ende!!!!!!-------########

        ######-------!!!!!!Aufgabe!!!!!!-------------########
        # Hier sollten die korrekten Anfangs und Endwerte für den flachen Ausgang berechnet werden
        # Achtung: Hier sollten alle Werte relativ zum Arbeitspunkt angegeben werden
        #y = Crnf * x + D * u
        # je nach Kroneckerindizes sind eta_1,k+1 oder eta_1,k+1 gleich eta_1,k bzw. eta_2,k
        xa, ua_equi = Heli().equilibrium(ya)
        xb, ub_equi = Heli().equilibrium(yb)
        C = np.zeros((2, 2))
        if kronecker == [1, 2]:
            C = np.array([[Crnf[0][0], Crnf[0][1]+Crnf[0][2]],
                          [Crnf[1][0], Crnf[1][1]+Crnf[1][2]]], dtype="complex_")

        elif kronecker == [2, 1]:
            C = np.array([[Crnf[0][0]+Crnf[0][1], Crnf[0][2]],
                          [Crnf[1][0]+Crnf[1][1], Crnf[1][2]]], dtype="complex_")

        self.eta_a = np.linalg.inv(
            C) @ (ya_rel-np.dot(self.linearized_system.D, ua_equi - self.linearized_system.u_equi))
        self.eta_b = np.linalg.inv(
            C) @ (yb_rel-np.dot(self.linearized_system.D, ub_equi - self.linearized_system.u_equi))
        ######-------!!!!!!Aufgabe Ende!!!!!!-------########

    def flat_output(self, k, index, shift=0):
        """Berechnet zeitdiskret die Trajektorie des flachen Ausgangs

        Parameter:
        ----------
        # k: diskrete Zeitpunkte als Vektor oder Skalar
        # shift: Linksverschiebung der Trajektorie
        # index: Komponente des flachen Ausgangs"""

        ######-------!!!!!!Aufgabe!!!!!!-------------########
        # Hier sollten die korrekten Werte für die um "shift" nach links verschobene Trajektorie des flachen Ausgangs zurückgegeben werden
        tau = (k+shift)/(self.N)

        return self.eta_a[index] + (self.eta_b[index] - self.eta_a[index]) * poly_transition(tau, 0, self.maxderi[index])
        ######-------!!!!!!Aufgabe Ende!!!!!!-------########

    # Zustandstrajektorie
    def state(self, k):
        kv = np.atleast_1d(k)
        dim_u = np.size(self.linearized_system.u_equi)
        dim_x = np.size(self.linearized_system.x_equi)
        dim_k = np.size(k)
        ######-------!!!!!!Aufgabe!!!!!!-------------########
        # analog zu state im kont.
        state = np.zeros((dim_x, dim_k))

        eta = list()
        for index in range(dim_u):
            eta = eta+[self.flat_output(kv, index, shift)
                       for shift in range(self.kronecker[index])]
        xrnf = np.vstack(eta)
        x = np.linalg.inv(self.Q) @ xrnf

        state = np.transpose(x.transpose() + self.linearized_system.x_equi)

        if (np.isscalar(k)):
            state = state[:, 0]
        ######-------!!!!!!Aufgabe Ende!!!!!!-------########
        return state

    # Ausgangstrajektorie
    def output(self, k):
        kv = np.atleast_1d(k)
        dim_y = np.size(self.linearized_system.y_equi)
        dim_x = np.size(self.linearized_system.x_equi)
        dim_u = np.size(self.linearized_system.u_equi)

        x_abs = self.state(kv)
        u_abs = self.input(kv)
        x_rel = x_abs-self.linearized_system.x_equi.reshape((dim_x, 1))
        u_rel = u_abs-self.linearized_system.u_equi.reshape((dim_u, 1))
        y_rel = self.linearized_system.C@x_rel+self.linearized_system.D@u_rel
        y_abs = y_rel+self.linearized_system.y_equi.reshape((dim_y, 1))
        if (np.isscalar(k)):
            y_abs = result[:, 0]
        return y_abs

    # Eingangstrajektorie
    def input(self, k):
        kv = np.atleast_1d(k)
        dim_u = np.size(self.linearized_system.u_equi)
        dim_k = np.size(kv)
        ######-------!!!!!!Aufgabe!!!!!!-------------########
        # analog zu input im kont.
        eta = list()
        for index in range(dim_u):
            eta = eta+[self.flat_output(kv, index, shift)
                       for shift in range(self.kronecker[index])]
        xrnf = np.vstack(eta)
        v = -self.A_rnf[self.kronecker.cumsum()-1, :]@xrnf
        for jj in range(self.kronecker.shape[0]):
            v[jj, :] += self.flat_output(kv, jj, self.kronecker[jj])
        input = (np.linalg.inv(self.M)@v) + \
            self.linearized_system.u_equi.reshape((dim_u, 1))
        ######-------!!!!!!Aufgabe Ende!!!!!!-------########
        if (np.isscalar(k)):
            input = input[:, 0]
        return input


def plot_results(t, x, u, y):
    plt.figure(figsize=(15, 7))
    plt.subplot(2, 2, 1, ylabel="Winkel $\\epsilon$ in Grad")
    plt.grid()
    leg = ["Soll", "Ist"]
    for v in y:
        plt.plot(t, v[0, :]/np.pi*180)
    plt.legend(leg)
    plt.subplot(2, 2, 2, ylabel="Winkelgeschwindigkeit $\\dot{\\alpha}$")
    plt.grid()
    for v in y:
        plt.plot(t, v[1, :]/np.pi*180)
    leg = ["Soll", "Ist"]
    plt.subplot(2, 2, 3, ylabel="Eingang 1 1/s")
    plt.grid()
    for v in u:
        plt.plot(t, v[0, :])
    plt.legend(leg)
    plt.subplot(2, 2, 4, ylabel="Eingang 2 1/s")
    plt.grid()
    for v in u:
        plt.plot(t, v[1, :])
    plt.legend(leg)
