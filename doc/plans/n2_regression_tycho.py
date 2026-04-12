import numpy as np, tychopy as typy
vf=typy.VectorFunctions; oc=typy.OptimalControl; Args=vf.Arguments
class TB(oc.ode_x_u.ode):
    def __init__(s,mu,a=False):
        Uv=3 if a else 0
        args=oc.ODEArguments(6,Uv); r=args.head3(); v=args.segment3(3)
        g=r.normalized_power3()*(-mu)
        acc=g+args.tail3()*a if a else g
        super().__init__(vf.stack([v,acc]),6,Uv)
def cs(r,td):
    v=np.sqrt(1/r); t=np.deg2rad(td); X=np.zeros(7)
    X[0]=np.cos(t)*r; X[1]=np.sin(t)*r; X[3]=-np.sin(t)*v; X[4]=np.cos(t)*v
    return X
def traj(r,td,tf,n):
    ode=TB(1,False); integ=ode.integrator(0.01)
    return [np.concatenate([T,[0.01]*3]) for T in integ.integrate_dense(cs(r,td),tf,n)]
n=2; LT=0.01; N=75
TIG=[traj(1,th,2*np.pi,300) for th in np.linspace(0,20,n)]
SP=TIG[0][-1][:7]
ocp=oc.OptimalControlProblem(); ode=TB(1,LT)
for T in TIG:
    ph=ode.phase('LGL5'); ph.set_traj(T,N); ph.set_control_mode('BlockConstant')
    ph.add_value_lock('Front',range(7))
    ph.add_lu_norm_bound('Path',[7,8,9],0.01,1.0,1)
    ph.add_delta_time_objective(1.0)
    ocp.add_phase(ph)
ocp.set_link_params(SP)
LF=Args(14).head(7)-Args(14).tail(7)
for i in range(n):
    ocp.add_link_equal_con(LF,[(i,'Back',range(7),[],[])],range(7))
ocp.add_link_param_equal_con(Args(6).head3().dot(Args(6).tail3()),range(6))
IS=[cs(1,th) for th in np.linspace(0,20,n)]
for i,ph in enumerate(ocp.phases):
    ph.sub_variables('Front',range(7),IS[i][:7])
ocp.optimizer.print_level=0
ocp.solve(); f=ocp.optimize()
lp=ocp.return_link_params()
print(f'RESULT n={n} flag={int(f)} tf_nd={lp[6]:.4f} pos_norm={float(np.linalg.norm(lp[:3])):.4f}')
