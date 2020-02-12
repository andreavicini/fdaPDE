#ifndef __SOLVER_HPP__
#define __SOLVER_HPP__

#define JOB_INIT -1
#define JOB_END -2
#define USE_COMM_WORLD -987654

#include "fdaPDE.h"
#include "../inst/include/dmumps_c.h"

//!  A Linear System QR solver class
/*!
 * This class gives offers a standard interface to the QR resolutor for dense matrices
*/
class QR{
	public:
	static void solve(const MatrixXr & A, const VectorXr & b,VectorXr &x){x=A.householderQr().solve(b);};
};

//!  A Linear System LU Partial Pivoting solver class
/*!
 * This class gives offers a standard interface to the LU Partial Pivoting resolutor for dense matrices.
 * OBS: The matrix should be invertible.
*/
class LUPV{
	public:
	static void solve(MatrixXr const & A, VectorXr const & b,VectorXr &x){x=A.partialPivLu().solve(b);};
};

//!  A Linear System LDLT solver class
/*!
 * This class gives offers a standard interface to the LDLT resolutor for dense matrices.
 * OBS: The matrix should be symmetric and SDP.
*/
class Symmetric{
	public:
	static void solve(MatrixXr const & A, VectorXr const & b,VectorXr &x){x=A.ldlt().solve(b);};
};

//!  A Linear System Cholesky solver class
/*!
 * This class gives offers a standard interface to the Cholesky resolutor for dense matrices.
 * OBS: The matrix should be symmetric and SDP, faster and more stable than others.
*/
class Cholesky{
	public:
	static void solve(MatrixXr const & A, VectorXr const & b,VectorXr &x){x=A.ldlt().solve(b);};
};

//!  A Linear System LU sparse solver class
/*!
 * This class gives offers a standard interface to the LU resolutor for sparse matrices.
*/
class SpLU{
	public:
	static void solve(SpMat const & A, VectorXr const & b, VectorXr &x )
	{
		Eigen::SparseLU<SpMat> solver;
		solver.compute(A);
		if(solver.info()!=Eigen::Success){
		//std::cerr<<"Decomposition failed!"<<std::endl;
		}
		x=solver.solve(b);
		if(solver.info()!=Eigen::Success)
		{
		//std::cerr<<"solving failed!"<<std::endl;
		}
	};
};

//!  A Linear System QR sparse solver class
/*!
 * This class gives offers a standard interface to the QR resolutor for sparse matrices.
*/
class SpQR{
	public:
	static void solve(SpMat const & A, VectorXr const & b, VectorXr &x )
	{
		Eigen::SparseQR<SpMat,Eigen::COLAMDOrdering<int> > solver;
		solver.compute(A);
		if(solver.info()!=Eigen::Success){
		//std::cerr<<"Decomposition failed!"<<std::endl;
		}
		x=solver.solve(b);
		if(solver.info()!=Eigen::Success)
		{
		//std::cerr<<"solving failed!"<<std::endl;
		}
	};
};

//!  A Linear System Cholesky sparse solver class
/*!
 * This class gives offers a standard interface to the Cholesky resolutor for sparse matrices.
*/
class SpCholesky{
	public:
	static void solve(SpMat const & A, VectorXr const & b, VectorXr &x )
	{
		Eigen::SimplicialLDLT<SpMat> solver;
		solver.compute(A);
		if(solver.info()!=Eigen::Success){
		//std::cerr<<"Decomposition failed!"<<std::endl;
		}
		x=solver.solve(b);
		if(solver.info()!=Eigen::Success)
		{
		//std::cerr<<"solving failed!"<<std::endl;
		}
	};
};

//!  A Linear System Conjugate Gradient sparse solver class
/*!
 * This class gives offers a standard interface to the Conjugate Gradient resolutor for sparse matrices.
*/
class SpConjGrad{
	public:
	static void solve(SpMat const & A, VectorXr const & b, VectorXr &x )
	{
		Eigen::ConjugateGradient<SpMat> solver;
		solver.compute(A);
		if(solver.info()!=Eigen::Success){
		//std::cerr<<"Decomposition failed!"<<std::endl;
		}
		x=solver.solve(b);
		if(solver.info()!=Eigen::Success)
		{
		//std::cerr<<"solving failed!"<<std::endl;
		}
	};
};

//!  A Linear System BiConjugate Gradient stabilized sparse solver class
/*!
 * This class gives offers a standard interface to the BiConjugate Gradient stabilized resolutor for sparse matrices.
*/

class BiCGSTAB{
	public:
	static void solve(SpMat const & A, VectorXr const & b, VectorXr &x )
	{
		Eigen::BiCGSTAB<SpMat> solver;
		solver.compute(A);
		if(solver.info()!=Eigen::Success){
		//std::cerr<<"Decomposition failed!"<<std::endl;
		}
		x=solver.solve(b);
		if(solver.info()!=Eigen::Success)
		{
		//std::cerr<<"solving failed!"<<std::endl;
		}
	};
};

//!  A Linear System BiConjugate Gradient stabilized with Incomplete LUT preconditioner sparse solver class
/*!
 * This class gives offers a standard interface to the BiConjugate Gradient stabilized BiConjugate Gradient stabilized with Incomplete LUT preconditioner resolutor for sparse matrices.
*/

class BiCGSTABILUT{
	public:
	static void solve(SpMat const & A, VectorXr const & b, VectorXr &x )
	{
		Eigen::BiCGSTAB<SpMat,Eigen::IncompleteLUT<Real>> solver;
		solver.compute(A);
		if(solver.info()!=Eigen::Success){
		//std::cerr<<"Decomposition failed!"<<std::endl;
		}
		x=solver.solve(b);
		if(solver.info()!=Eigen::Success)
		{
		//std::cerr<<"solving failed!"<<std::endl;
		}
	};
};


class Mumps{
	public:
		template<typename Derived1,typename Derived2>
    static void solve(SpMat const & A,const Eigen::MatrixBase<Derived1> & b, Eigen::MatrixBase<Derived2> & x )
	{

		const Real *values = A.valuePtr();
		const UInt *inner = A.innerIndexPtr();
		const UInt *outer = A.outerIndexPtr();

		UInt n = A.cols();

		std::vector<int> irn;
		std::vector<int> jcn;
		std::vector<double> a;

		for (int j=0; j<A.outerSize(); ++j)
		{
			for (SpMat::InnerIterator it(A,j); it; ++it)
			{
				if(it.col()>=it.row())
				{
					irn.push_back(it.row()+1);
					jcn.push_back(it.col()+1);
					a.push_back(it.value());
				}
			}
		}

    DMUMPS_STRUC_C id;

		//Real *rhs = b.array();
    Real rhs[b.rows()*b.cols()];
		for(UInt j = 0; j < b.cols(); ++j)
			for(UInt i = 0; i < b.rows(); ++i)
				rhs[i+j*b.rows()] = b(i,j);
    /* Initialize a MUMPS instance. Use MPI_COMM_WORLD */
    id.job=JOB_INIT; id.par=1; id.sym=2; id.comm_fortran=USE_COMM_WORLD;
    dmumps_c(&id);
    /* Define the problem on the host */

		id.n = n; id.nz =irn.size(); id.irn=irn.data(); id.jcn=jcn.data();
		id.a = a.data();
		id.lrhs = b.rows(); id.nrhs = b.cols(); id.rhs = rhs;

    #define ICNTL(I) icntl[(I)-1] /* macro s.t. indices match documentation */
    /* No outputs */
    id.ICNTL(1)=-1; id.ICNTL(2)=-1; id.ICNTL(3)=-1; id.ICNTL(4)=0;
		id.ICNTL(14)=1000;
    /* Call the MUMPS package. */
    id.job=6;
    dmumps_c(&id);
    id.job=JOB_END; dmumps_c(&id); /* Terminate instance */

		for(UInt j = 0; j < b.cols(); ++j)
			for(UInt i = 0; i < b.rows(); ++i)
				x(i,j) = rhs[i+j*b.rows()];

   };
};


#endif
