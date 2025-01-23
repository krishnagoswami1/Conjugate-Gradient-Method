import numpy as np
import time
import streamlit as st
st.set_page_config(layout="wide", page_title = "CG Method",page_icon= ":triangular-ruler:")

def conjugate_gradient_solver(A , b, x0=None, tol=1e-10,max_iter=500):
    init_time = time.time()
    # A is (n,n) matrix
    #b is (n,) vector
    #x0 is the initial guess, (n,) vector
    #r,v,x are (n,) vector
    #s,t are scalars
    n=len(b)
    if x0 == None:
        x0 = np.zeros((n,))

    x=x0.copy()



    for i in range(max_iter):
        r0 = b - A @ x
        if i==0:
            v = r0.copy()

        t= (v @ r0) / (v @ (A @ v))
        x = x + t * v

        r1 = b - A @ x
        #check tolerance
        if (r1 @ r1) < tol:
            break

        s = (r1 @ r1)/(r0 @ r0)
        v = r1 + s * v

    return x, i+1,time.time() - init_time

#UI HERE
st.title("Conjugate Method Solver")
st.sidebar.title("Configuration")
n = st.sidebar.slider("Size of matrix:", min_value = 2, max_value = 20, step=1)

input_type = st.sidebar.radio("",options=["Random Input", "Manual Input"])
A =[]
b=[]
if input_type== "Random Input":
    A = np.random.randint(0,20,size=n*n).reshape(n,n)
    A = A @ A.T
    b = np.random.randint(0,50,size=n)
elif input_type == "Manual Input":
    tab1, tab2 = st.sidebar.tabs(['Matrix', 'vector'])
    with tab1:
        st.write("Enter the matrix A (symmetric and positive definite)")
        for i in range(n):
            for j in range(n):

                A.append(st.number_input(f"A[{i}][{j}]=",value =0.00,step=0.01,format = "%.2f"))

    with tab2:
        st.write("Enter the vector :")
        for i in range(n):
            b.append(st.number_input(f"b[{i}]=",value = 0.0, step =0.01, format="%.2f"))

    A = np.array(A).reshape(n,n)
    b=np.array(b).reshape(n,)

def print_linear_equations(A, b):
    """
    Generate a string representation of the linear system Ax = b

    Parameters:
    A: Coefficient matrix
    b: Right-hand side vector

    Returns:
    str: Formatted system of linear equations
    """
    n = len(b)
    equations = []

    for i in range(n):
        # Build the equation for each row
        terms = []
        for j in range(n):
            # Only add non-zero terms
            if A[i][j] != 0:
                # Handle coefficient 1 and -1 specially
                if A[i][j] == 1:
                    term = f' x<sub>{j}</sub> '
                elif A[i][j] == -1:
                    term = f' - x<sub>{j}</sub> '
                else:
                    term = f' {A[i][j]} x<sub>{j}</sub> '
                terms.append(term)

        # Join the terms with ' + '
        equation_str = ' + '.join(terms).replace('+ -', '- ')
        equation_str += f' = {b[i]}'
        equations.append(equation_str)

    return '\n'.join(equations)

if st.button("Solve"):
    eqs= print_linear_equations(A,b)
    for line in eqs.split("\n"):
        st.markdown(line, unsafe_allow_html = True)
    x_solution, num_iter, time_taken = conjugate_gradient_solver(A, b)

    st.write(f"Solution is ")
    for i,val in enumerate(np.round(x_solution,5)):
        st.markdown(f"x<sub>{i}</sub> = {val}",unsafe_allow_html = True)
    st.write(f"Number of iterations taken: ", num_iter)
    st.write(f"Time taken: ", time_taken)