#include <omp.h>
#include<gsl/gsl_rng.h>
#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include</home/joyjit/mt19937ar.h>
#include</home/joyjit/mt19937ar.c>

/* This prog simulates a system of monodispersed long rectangles of size m x mk with no intersection
 * allowed on a two dimensional lattice in grand canonical ensemble. It is a montecarlo simulation 
 * with heat bath dynamics where all the horizontal rectangles are evaporated and
 * redeposited, then vertical rectangles and so on. In addition to the evaporation-deposition 
 * move, it alos flips a block of size mk x mk containing m vertical rectangles to horizontal
 * ones and vice versa.This additional "flip" move expedites the equilibration.

/* parallelized using openMP*/

#define K 9 /* mk = 9; k=3 */
#define L 648	/*Number of sites in one direction, should be even*/
#define N (L*L) /* The total number of sites */
#define NN 2 /*Number of nearest neighbours in forward direction */
#define T_eq 5000000	/*# of MC steps for equilibration*/
#define BLOCKS 10 /*Number of blocks in which average is taken*/
#define AVE 30000000 /* # of readings in each block */
#define PSTART 0.7 /*starting probability*/
#define PDIFF 0.1 /*increment in probability*/
#define PEND (PSTART+0.5*PDIFF) /*ending probability */
#define GAP2 20000 /*GAP after which output is written during equilibration */
#define INITIAL_FLAG 0 /*0: all empty 1: all filled */
#define BINS 10001 /*Keep it odd*/
#define BINSIZE (2.0/(1.0*BINS))
#define BINSIZER (1.0/(1.0*BINS))
#define BINSIZEQ 1000
#define BINSQ (BINSIZEQ+1)
#define width 3
#define nthreads 8
#define pi (4*atan(1))
#define sub 3
#define nsize (K*K)
#define nturns (N/nsize)
#define LBS (L/K)
#define rc (2*sub)
#define AR (K/width)

int lat[N],ln[N],rn[N],bn[N],tn[N];
int h[sub][sub],h0[rc];
int starth,finalh,endh,startv,finalv,endv,taskid,nov,noh;
double binq[BINS],periodic[K],acceptance[L+1],mu;
double meanrho[BLOCKS],rho2[BLOCKS],fluc_r[BLOCKS],fluc_Q[BLOCKS],fluc_QN[BLOCKS];
double meanm1[BLOCKS],meanm2[BLOCKS],meanm4[BLOCKS],qnem1[BLOCKS],qnem2[BLOCKS],qnem4[BLOCKS];
double Q_1[BLOCKS],Q_2[BLOCKS],Q_4[BLOCKS],Rho_2[BLOCKS],Rho[BLOCKS],QN1[BLOCKS],QN2[BLOCKS],QN4[BLOCKS];
double mass[BINS],factor[BINS],bincount[BINS],count[BINS],massr[BINS],factorr[BINS],bincountr[BINS],countr[BINS];
double sum_tmpq,sum_tmpr,num_tmp;
char outfile1[100],outfile2[100],outfile3[100],outfile4[100],outfile5[100];
char readfile1[100],readfile2[100];
long seedval[nthreads+1],seedval1;
const gsl_rng_type *T;
gsl_rng *r[nthreads+1];
#pragma omp threadprivate(taskid,starth,finalh,endh,startv,finalv,endv)


int find_site(int x, int y)
{
	/* Given x, y it gives site index */
	x=(x+L) % L;
	y=(y+L) % L;
	return x+y*L;
}


void neighbour()
{
	int i,j,x,y;
	for(y=0;y<L;y++)
	{
		for(x=0;x<L;x++)
		{
			i=find_site(x,y);
			j=find_site(x-1,y); ln[i]=j;
			j=find_site(x+1,y); rn[i]=j;
			j=find_site(x,y-1); bn[i]=j;
			j=find_site(x,y+1); tn[i]=j;
		}
	}
}

void initialize()
{
	/* initializes nbr list and output file names */
	int i,j,x,y;
	double tmp,tmp1;

	if(INITIAL_FLAG==0)
		sprintf(outfile1,"rec_emptyK%dL%dM%5.4lf_omp",K,L,mu);
	if(INITIAL_FLAG==1)
		sprintf(outfile1,"rec_filledK%dL%dM%5.4lf_omp",K,L,mu);
	/*initializing bin parameters*/

	for(j=0;j<BINS;j++)
	{
		bincount[j]=0; bincountr[j]=0; countr[j]=0;
		mass[j]=0; massr[j]=0; count[j]=0;
	}
	for(j=-L*L;j<=L*L;j=j+width*K)
	{
		tmp=(1.0*j)/(1.0*N);
		i=floor((tmp+1.0)/BINSIZE);
		if(j==-N)
			i=0;
		if(j==N)
			i=BINS-1;
		bincount[i]++;
		mass[i]=mass[i]+tmp;
	}
	tmp1=(width*K*1.0)/(2.0*N+1.0*width*K);
	for(j=0;j<BINS;j++)
	{
		mass[j]=mass[j]/bincount[j];
		factor[j]=tmp1*bincount[j];
	}

	for(j=0;j<=L*L;j=j+width*K)
	{
		tmp=(1.0*j)/(1.0*N);
		i=floor(tmp/BINSIZER);
		if(j==N)
			i=BINS-1;
		bincountr[i]++;
		massr[i]=massr[i]+tmp;
	}
	tmp1=(width*K*1.0)/(1.0*N+1.0*width*K);
	for(j=0;j<BINS;j++)
	{
		massr[j]=massr[j]/bincountr[j];
		factorr[j]=tmp1*bincountr[j];
	}

	for(i=0;i<BINSQ;i++)
		binq[i] = 0.0;	
}

void lat_init()
{
	/* Initializes quantities that have to initialized for every value
	 * of probability p */
	int i;
	double x;
	FILE *fp;
	for(i=0;i<BLOCKS;i++)
	{
		meanrho[i]=0; rho2[i]=0; meanm1[i]=0; meanm2[i]=0; meanm4[i]=0; qnem1[i]=0; qnem2[i]=0; qnem4[i]=0; 
		Rho[i]=0; Rho_2[i]=0; Q_1[i]=0; Q_2[i]=0; Q_4[i]=0; fluc_r[i]=0; fluc_Q[i]=0; QN1[i]=0; QN2[i]=0; QN4[i]=0; fluc_QN[i]=0; 
	}
	if((L % K !=0)|| (L % 2) !=0)
	{
		printf("ERROR IN DIVISIBILITY\n");
		exit(0);
	}
	
	if(INITIAL_FLAG==0)
	{	
		sprintf(outfile2,"rec_emptyK%dL%dM%5.4lfomp_t",K,L,mu);
		sprintf(outfile3,"rec_emptyK%dL%dM%5.4lfomp_p_r",K,L,mu);
		sprintf(outfile4,"rec_emptyK%dL%dM%5.4lfomp_p_q",K,L,mu);
		sprintf(outfile5,"rec_emptyK%dL%dM%5.4lfomp_p_qnm",K,L,mu);
	}
	if(INITIAL_FLAG==1)
	{
		sprintf(outfile2,"rec_filledK%dL%dM%5.4lfomp_t",K,L,mu);
		sprintf(outfile3,"rec_filledK%dL%dM%5.4lfomp_p_r",K,L,mu);
		sprintf(outfile4,"rec_filledK%dL%dM%5.4lfomp_p_q",K,L,mu);
		sprintf(outfile5,"rec_filledK%dL%dM%5.4lfomp_p_qnm",K,L,mu);
	}
	fp=fopen(outfile2,"w");
	fprintf(fp,"#t rho abs(m) m\n");
	fclose(fp);

	sprintf(readfile1,"acceptprobK%dL%dM%5.4lf",K,L,mu);
	fp=fopen(readfile1,"r");
	if(fp==NULL)
	{
		printf("The FILE [%s] DOES NOT EXIST\n",readfile1);
		exit(0);
	}
	while(fscanf(fp,"%d%lf",&i,&x)!=EOF)
		acceptance[i]=x;
	fclose(fp);
	if(i!=L)
	{
		printf("Error in FILE [%s]\n",readfile1);
		exit(0);
	}

	sprintf(readfile2,"periodicprobK%dL%dM%5.4lf",K,L,mu);
	fp=fopen(readfile2,"r");
	if(fp==NULL)
	{
		printf("The FILE [%s] DOES NOT EXIST\n",readfile2);
		exit(0);
	}
	while(fscanf(fp,"%d%lf",&i,&x)!=EOF)
		periodic[i]=x;
	fclose(fp);
	if(i!=K-1)
	{
		printf("Error in FILE [%s]\n",readfile2);
		exit(0);
	}

}

void remove_hor_1()
{
	/* removes all horizontal kmers */
	int i;

	#pragma omp for private(i) 
	for(i=0;i<N;i++)
		if((lat[i]==1)||(lat[i]==-1))
			lat[i]=0;
}
void remove_hor_2()
{
	/* removes all horizontal kmers */
	int i;

	#pragma omp for private(i) 
	for(i=0;i<N;i++)
		if((lat[i]==2)||(lat[i]==-2))
			lat[i]=0;
}
void remove_hor_3()
{
	/* removes all horizontal kmers */
	int i;

	#pragma omp for private(i) 
	for(i=0;i<N;i++)
		if((lat[i]==3)||(lat[i]==-3))
			lat[i]=0;
}
void remove_ver_1()
{
	/* removes all vertical kmers */
	int i;

	#pragma omp for private(i) 
	for(i=0;i<N;i++)
		if((lat[i]==4)||(lat[i]==-4))
			lat[i]=0;
}
void remove_ver_2()
{
	/* removes all vertical kmers */
	int i;

	#pragma omp for private(i) 
	for(i=0;i<N;i++)
		if((lat[i]==5)||(lat[i]==-5))
			lat[i]=0;
}
void remove_ver_3()
{
	/* removes all vertical kmers */
	int i;

	#pragma omp for private(i) 
	for(i=0;i<N;i++)
		if((lat[i]==6)||(lat[i]==-6))
			lat[i]=0;
}

void remove_xmer(int i)
{
	/* removes an xmer with head at i */
	int j,W,pos;

	for(W=0;W<width;W++)
	{
		pos=i;	
		for(j=0;j<K;j++)
		{
			lat[i]=0;
			i=rn[i];
		}
		i=tn[pos];
	}
}

void remove_ymer(int i)
{
	/* removes an xmer with head at i */
	int j,W,pos;
	for(W=0;W<width;W++)
	{
		pos=i;	
		for(j=0;j<K;j++)
		{
			lat[i]=0;
			i=tn[i];
		}
		i=rn[pos];
	}
}

void deposit_hor_1(int i)
{
	/*puts a horizontal kmer with head at i*/

	int j,W,pos,hd;
	hd=i;
	for(W=0;W<width;W++)
	{
		pos=i;
		for(j=0;j<K;j++)
		{
			lat[i]=1;
			i=rn[i];		
		}
		i=tn[pos];
	}

	lat[hd]=-1;
}
void deposit_hor_2(int i)
{
	/*puts a horizontal kmer with head at i*/
	int j,W,pos,hd;
	hd=i;
	for(W=0;W<width;W++)
	{
		pos=i;
		for(j=0;j<K;j++)
		{
			lat[i]=2;
			i=rn[i];		
		}
		i=tn[pos];
	}

	lat[hd]=-2;
}
void deposit_hor_3(int i)
{
	/*puts a horizontal kmer with head at i*/
	int j,W,pos,hd;
	hd=i;
	for(W=0;W<width;W++)
	{
		pos=i;
		for(j=0;j<K;j++)
		{
			lat[i]=3;
			i=rn[i];		
		}
		i=tn[pos];
	}

	lat[hd]=-3;
}
void deposit_ver_1(int i)
{
	/*puts a vertical kmer with head at i*/

	int j,W,pos,hd;
	hd=i;
	for(W=0;W<width;W++)
	{
		pos=i;
		for(j=0;j<K;j++)
		{
			lat[i]=4;
			i=tn[i];		
		}
		i=rn[pos];
	}

	lat[hd]=-4;
}
void deposit_ver_2(int i)
{
	/*puts a vertical kmer with head at i*/

	int j,W,pos,hd;
	hd=i;
	for(W=0;W<width;W++)
	{
		pos=i;
		for(j=0;j<K;j++)
		{
			lat[i]=5;
			i=tn[i];		
		}
		i=rn[pos];
	}

	lat[hd]=-5;
}
void deposit_ver_3(int i)
{
	/*puts a vertical kmer with head at i*/
	int j,W,pos,hd;
	hd=i;
	for(W=0;W<width;W++)
	{
		pos=i;
		for(j=0;j<K;j++)
		{
			lat[i]=6;
			i=tn[i];		
		}
		i=rn[pos];
	}

	lat[hd]=-6;
}

void fill_periodic_hor_1(int j)
{
	/*Given an empty horizontal line, the function checks the
	 * occupation probability of first K-1 sites */

	int i,k,W1,pos1,starth_1,tmp;

	starth=j; finalh=ln[j];
	for(i=0;i<K-1;i++)
	{
		tmp=starth;
		if(gsl_rng_uniform(r[taskid]) < periodic[i])
		{
			for(W1=0;W1<width;W1++)
			{
				pos1=starth;				
				for(k=0;k<K;k++)
				{
					lat[starth]=1;
					starth=rn[starth];
				}
				starth_1=starth;
				starth=tn[pos1];
			}
			lat[tmp]=-1;
			starth=bn[bn[starth_1]];
			return;
		}
		else
		{
			finalh=starth;
			starth=rn[starth];
		}
	}
	return;
}
void fill_periodic_hor_2(int j)
{
	/*Given an empty horizontal line, the function checks the
	 * occupation probability of first K-1 sites */

	int i,k,W1,pos1,starth_1,tmp;

	starth=j;finalh=ln[j];
	for(i=0;i<K-1;i++)
	{
		tmp=starth;
		if(gsl_rng_uniform(r[taskid]) < periodic[i])
		{
			for(W1=0;W1<width;W1++)
			{
				pos1=starth;				
				for(k=0;k<K;k++)
				{
					lat[starth]=2;
					starth=rn[starth];
				}
				starth_1=starth;
				starth=tn[pos1];
			}
			lat[tmp]=-2;
			starth=bn[bn[starth_1]];
			return;
		}
		else
		{
			finalh=starth;
			starth=rn[starth];
		}
	}
	return;
}
void fill_periodic_hor_3(int j)
{
	/*Given an empty horizontal line, the function checks the
	 * occupation probability of first K-1 sites */

	int i,k,W1,pos1,starth_1,tmp;

	starth=j;finalh=ln[j];
	for(i=0;i<K-1;i++)
	{
		tmp=starth;
		if(gsl_rng_uniform(r[taskid]) < periodic[i])
		{
			for(W1=0;W1<width;W1++)
			{
				pos1=starth;				
				for(k=0;k<K;k++)
				{
					lat[starth]=3;
					starth=rn[starth];
				}
				starth_1=starth;
				starth=tn[pos1];
			}
			lat[tmp]=-3;
			starth=bn[bn[starth_1]];
			return;
		}
		else
		{
			finalh=starth;
			starth=rn[starth];
		}
	}
	return;
}

void fill_periodic_ver_1(int j)
{
	/*Given an empty vertical line, the function checks the
	 * occupation probability of first K-1 sites */

	int i,k,W1,pos1,startv_1,tmp;

	startv=j;finalv=bn[j];
	for(i=0;i<K-1;i++)
	{
		tmp=startv;
		if(gsl_rng_uniform(r[taskid]) < periodic[i])
		{
			for(W1=0;W1<width;W1++)
			{
				pos1=startv;				
				for(k=0;k<K;k++)
				{
					lat[startv]=4;
					startv=tn[startv];
				}
				startv_1=startv;
				startv=rn[pos1];
			}
			lat[tmp]=-4;
			startv=ln[ln[startv_1]];
			return;
		}
		else
		{
			finalv=startv;
			startv=tn[startv];
		}
	}
	return;
}
void fill_periodic_ver_2(int j)
{
	/*Given an empty vertical line, the function checks the
	 * occupation probability of first K-1 sites */

	int i,k,W1,pos1,startv_1,tmp;

	startv=j;finalv=bn[j];
	for(i=0;i<K-1;i++)
	{
		tmp=startv;
		if(gsl_rng_uniform(r[taskid]) < periodic[i])
		{
			for(W1=0;W1<width;W1++)
			{
				pos1=startv;				
				for(k=0;k<K;k++)
				{
					lat[startv]=5;
					startv=tn[startv];
				}
				startv_1=startv;
				startv=rn[pos1];
			}
			lat[tmp]=-5;
			startv=ln[ln[startv_1]];
			return;
		}
		else
		{
			finalv=startv;
			startv=tn[startv];
		}
	}
	return;
}
void fill_periodic_ver_3(int j)
{
	/*Given an empty vertical line, the function checks the
	 * occupation probability of first K-1 sites */

	int i,k,W1,pos1,startv_1,tmp;

	startv=j;finalv=bn[j];
	for(i=0;i<K-1;i++)
	{
		tmp=startv;
		if(gsl_rng_uniform(r[taskid]) < periodic[i])
		{
			for(W1=0;W1<width;W1++)
			{
				pos1=startv;				
				for(k=0;k<K;k++)
				{
					lat[startv]=6;
					startv=tn[startv];
				}
				startv_1=startv;
				startv=rn[pos1];
			}
			lat[tmp]=-6;
			startv=ln[ln[startv_1]];
			return;
		}
		else
		{
			finalv=startv;
			startv=tn[startv];
		}
	}
	return;
}

int find_starthfinalh_1(int row)
{
	/*in row 'row' finds out starth and finalh*/

	int i;

	starth=row*L; finalh=row*L;
	if(lat[starth]!=0 || lat[tn[starth]]!=0 || lat[tn[tn[starth]]]!=0)
	{
		while(lat[starth]!=0 || lat[tn[starth]]!=0 || lat[tn[tn[starth]]]!=0)
		{
			starth=rn[starth];
			if(starth==row*L)
				return 0;
		}
		while(lat[finalh]!=0 || lat[tn[finalh]]!=0 || lat[tn[tn[finalh]]]!=0)
			finalh=ln[finalh];
		return 1;
	}

	while(lat[ln[starth]]==0 && lat[tn[ln[starth]]]==0 && lat[tn[tn[ln[starth]]]]==0)
	{
		starth=ln[starth];
		if(starth==row*L)
		{
			fill_periodic_hor_1(starth);
			return 1;
		}
	}
	finalh=ln[starth];
	while(lat[finalh]!=0 || lat[tn[finalh]]!=0 || lat[tn[tn[finalh]]]!=0)
		finalh=ln[finalh];
	return 1;
}
int find_starthfinalh_2(int row)
{
	/*in row 'row' finds out starth and finalh*/

	int i;

	starth=row*L; finalh=row*L;
	if(lat[starth]!=0 || lat[tn[starth]]!=0 || lat[tn[tn[starth]]]!=0)
	{
		while(lat[starth]!=0 || lat[tn[starth]]!=0 ||lat[tn[tn[starth]]]!=0 )
		{	
			starth=rn[starth];
			if(starth==row*L)
				return 0;
		}
		while(lat[finalh]!=0 || lat[tn[finalh]]!=0 || lat[tn[tn[finalh]]]!=0)
			finalh=ln[finalh];
		return 1;
	}

	while(lat[ln[starth]]==0 && lat[tn[ln[starth]]]==0 && lat[tn[tn[ln[starth]]]]==0)
	{	
		starth=ln[starth];
		if(starth==row*L)
		{
			fill_periodic_hor_2(starth);
			return 1;
		}
	}
	finalh=ln[starth];
	while(lat[finalh]!=0 || lat[tn[finalh]]!=0 || lat[tn[tn[finalh]]]!=0 )
		finalh=ln[finalh];
	return 1;
}
int find_starthfinalh_3(int row)
{
	/*in row 'row' finds out starth and finalh*/

	int i;

	starth=row*L; finalh=row*L;
	if(lat[starth]!=0 || lat[tn[starth]]!=0 || lat[tn[tn[starth]]]!=0)
	{
		while(lat[starth]!=0 || lat[tn[starth]]!=0 || lat[tn[tn[starth]]]!=0)
		{	
			starth=rn[starth];
			if(starth==row*L)
				return 0;
		}
		while(lat[finalh]!=0 || lat[tn[finalh]]!=0 || lat[tn[tn[finalh]]]!=0)
			finalh=ln[finalh];
		return 1;
	}

	while(lat[ln[starth]]==0 && lat[tn[ln[starth]]]==0 && lat[tn[tn[ln[starth]]]]==0)
	{	
		starth=ln[starth];
		if(starth==row*L)
		{
			fill_periodic_hor_3(starth);
			return 1;
		}
	}
	finalh=ln[starth];
	while(lat[finalh]!=0 || lat[tn[finalh]]!=0 || lat[tn[tn[finalh]]]!=0)
		finalh=ln[finalh];
	return 1;
}

void fill_row_1(int row)
{
	/*fills row 'row' with horizontal kmers*/

	int i,len;

	if(find_starthfinalh_1(row)==0)
		return;
	do
	{
		endh=starth;len=1;
		while(lat[rn[endh]]==0 && lat[tn[rn[endh]]]==0 && lat[tn[tn[rn[endh]]]]==0)
		{			
			endh=rn[endh];
			len++;
			if(endh==finalh)		
				break;
		}
		while(len>=K)
		{
			if(gsl_rng_uniform(r[taskid]) < acceptance[len])
			{
				deposit_hor_1(starth);
				for(i=0;i<K;i++)
					starth=rn[starth];
				len=len-K;
			}
			else
			{
				starth=rn[starth];
				len--;
			}
		}
		if(endh==finalh)
			return;
		starth=rn[endh];
		while(lat[starth]!=0 || lat[tn[starth]]!=0 || lat[tn[tn[starth]]]!=0)
			starth=rn[starth];
	}
	while(endh!=finalh);
}
void fill_row_2(int row)
{
	/*fills row 'row' with horizontal kmers*/

	int i,len;

	if(find_starthfinalh_2(row)==0)
		return;
	do
	{
		endh=starth;len=1;
		while(lat[rn[endh]]==0 && lat[tn[rn[endh]]]==0 && lat[tn[tn[rn[endh]]]]==0)
		{	
			endh=rn[endh];
			len++;
			if(endh==finalh)	
				break;
		}
		while(len>=K)
		{	
			if(gsl_rng_uniform(r[taskid]) < acceptance[len])
			{
				deposit_hor_2(starth);
				for(i=0;i<K;i++)
					starth=rn[starth];
				len=len-K;
			}
			else
			{
				starth=rn[starth];
				len--;
			}
		}
		if(endh==finalh)
			return;
		starth=rn[endh];
		while(lat[starth]!=0 || lat[tn[starth]]!=0 || lat[tn[tn[starth]]]!=0)
			starth=rn[starth];
	}
	while(endh!=finalh);
}
void fill_row_3(int row)
{
	/*fills row 'row' with horizontal kmers*/

	int i,len;

	if(find_starthfinalh_3(row)==0)
		return;
	do
	{
		endh=starth;len=1;
		while(lat[rn[endh]]==0 && lat[tn[rn[endh]]]==0 && lat[tn[tn[rn[endh]]]]==0)
		{	
			endh=rn[endh];
			len++;
			if(endh==finalh)	
				break;
		}
		while(len>=K)
		{	
			if(gsl_rng_uniform(r[taskid]) < acceptance[len])
			{
				deposit_hor_3(starth);
				for(i=0;i<K;i++)
					starth=rn[starth];
				len=len-K;
			}
			else
			{
				starth=rn[starth];
				len--;
			}
		}
		if(endh==finalh)
			return;
		starth=rn[endh];
		while(lat[starth]!=0 || lat[tn[starth]]!=0 || lat[tn[tn[starth]]]!=0)
			starth=rn[starth];
	}
	while(endh!=finalh);
}
int find_startvfinalv_1(int col)
{
	/*in row 'row' finds out startv and finalv*/

	int i;

	startv=col; finalv=col;
	if(lat[startv]!=0 || lat[rn[startv]]!=0 || lat[rn[rn[startv]]]!=0)
	{
		while(lat[startv]!=0 || lat[rn[startv]]!=0 || lat[rn[rn[startv]]]!=0)
		{
			startv=tn[startv];
			if(startv==col)
				return 0;
		}
		while(lat[finalv]!=0 || lat[rn[finalv]]!=0 || lat[rn[rn[finalv]]]!=0)
			finalv=bn[finalv];
		return 1;
	}

	while(lat[bn[startv]]==0 && lat[rn[bn[startv]]]==0 && lat[rn[rn[bn[startv]]]]==0)
	{
		startv=bn[startv];
		if(startv==col)
		{
			fill_periodic_ver_1(startv);
			return 1;
		}
	}
	finalv=bn[startv];
	while(lat[finalv]!=0 || lat[rn[finalv]]!=0 || lat[rn[rn[finalv]]]!=0)
		finalv=bn[finalv];
	return 1;
}
int find_startvfinalv_2(int col)
{
	/*in row 'row' finds out startv and finalv*/

	int i;

	startv=col; finalv=col;
	if(lat[startv]!=0 || lat[rn[startv]]!=0 || lat[rn[rn[startv]]]!=0)
	{
		while(lat[startv]!=0 || lat[rn[startv]]!=0 || lat[rn[rn[startv]]]!=0)
		{
			startv=tn[startv];
			if(startv==col)
				return 0;
		}
		while(lat[finalv]!=0 || lat[rn[finalv]]!=0 || lat[rn[rn[finalv]]]!=0)
			finalv=bn[finalv];
		return 1;
	}

	while(lat[bn[startv]]==0 && lat[rn[bn[startv]]]==0 && lat[rn[rn[bn[startv]]]]==0)
	{
		startv=bn[startv];
		if(startv==col)
		{
			fill_periodic_ver_2(startv);
			return 1;
		}
	}
	finalv=bn[startv];
	while(lat[finalv]!=0 || lat[rn[finalv]]!=0 || lat[rn[rn[finalv]]]!=0)
		finalv=bn[finalv];
	return 1;
}
int find_startvfinalv_3(int col)
{
	/*in row 'row' finds out startv and finalv*/

	int i;

	startv=col; finalv=col;
	if(lat[startv]!=0 || lat[rn[startv]]!=0 || lat[rn[rn[startv]]]!=0)
	{
		while(lat[startv]!=0 || lat[rn[startv]]!=0 || lat[rn[rn[startv]]]!=0)
		{
			startv=tn[startv];
			if(startv==col)
				return 0;
		}
		while(lat[finalv]!=0 || lat[rn[finalv]]!=0 || lat[rn[rn[finalv]]]!=0)
			finalv=bn[finalv];
		return 1;
	}

	while(lat[bn[startv]]==0 && lat[rn[bn[startv]]]==0 && lat[rn[rn[bn[startv]]]]==0)
	{
		startv=bn[startv];
		if(startv==col)
		{
			fill_periodic_ver_3(startv);
			return 1;
		}
	}
	finalv=bn[startv];
	while(lat[finalv]!=0 || lat[rn[finalv]]!=0 || lat[rn[rn[finalv]]]!=0)
		finalv=bn[finalv];
	return 1;
}
void fill_col_1(int col)
{
	/*fills col 'col' with horizontal kmers*/

	int i,len;

	if(find_startvfinalv_1(col)==0)
		return;
	do
	{
		endv=startv;len=1;
		while(lat[tn[endv]]==0 && lat[rn[tn[endv]]]==0 && lat[rn[rn[tn[endv]]]]==0)
		{
			endv=tn[endv];
			len++;
			if(endv==finalv)
				break;
		}
		while(len>=K)
		{
			if(gsl_rng_uniform(r[taskid]) < acceptance[len])
			{
				deposit_ver_1(startv);
				for(i=0;i<K;i++)
					startv=tn[startv];
				len=len-K;
			}
			else
			{
				startv=tn[startv];
				len--;
			}
		}
		if(endv==finalv)
			return;
		startv=tn[endv];
		while(lat[startv]!=0 || lat[rn[startv]]!=0 || lat[rn[rn[startv]]]!=0)
			startv=tn[startv];
	}
	while(endv!=finalv);
}
void fill_col_2(int col)
{
	/*fills col 'col' with horizontal kmers*/

	int i,len;

	if(find_startvfinalv_2(col)==0)
		return;
	do
	{
		endv=startv;len=1;
		while(lat[tn[endv]]==0 && lat[rn[tn[endv]]]==0 && lat[rn[rn[tn[endv]]]]==0)
		{
			endv=tn[endv];
			len++;
			if(endv==finalv)
				break;
		}
		while(len>=K)
		{
			if(gsl_rng_uniform(r[taskid]) < acceptance[len])
			{
				deposit_ver_2(startv);
				for(i=0;i<K;i++)
					startv=tn[startv];
				len=len-K;
			}
			else
			{
				startv=tn[startv];
				len--;
			}
		}
		if(endv==finalv)
			return;
		startv=tn[endv];
		while(lat[startv]!=0 || lat[rn[startv]]!=0 || lat[rn[rn[startv]]]!=0)
			startv=tn[startv];
	}
	while(endv!=finalv);
}
void fill_col_3(int col)
{
	/*fills col 'col' with horizontal kmers*/

	int i,len;

	if(find_startvfinalv_3(col)==0)
		return;
	do
	{
		endv=startv;len=1;
		while(lat[tn[endv]]==0 && lat[rn[tn[endv]]]==0 && lat[rn[rn[tn[endv]]]]==0)
		{
			endv=tn[endv];
			len++;
			if(endv==finalv)
				break;
		}
		while(len>=K)
		{
			if(gsl_rng_uniform(r[taskid]) < acceptance[len])
			{
				deposit_ver_3(startv);
				for(i=0;i<K;i++)
					startv=tn[startv];
				len=len-K;
			}
			else
			{
				startv=tn[startv];
				len--;
			}
		}
		if(endv==finalv)
			return;
		startv=tn[endv];
		while(lat[startv]!=0 || lat[rn[startv]]!=0 || lat[rn[rn[startv]]]!=0)
			startv=tn[startv];
	}
	while(endv!=finalv);
}

void hortover_11(int i)
{
	int j,k,wi;

	k=i;
	for(j=0;j<AR;j++)
	{
		remove_xmer(k);
		for(wi=0;wi<width;wi++)
			k=tn[k];
	}
	k=i;
	for(j=0;j<AR;j++)
	{
		deposit_ver_1(k);
		for(wi=0;wi<width;wi++)
			k=rn[k];
	}
}

void hortover_12(int i)
{
	int j,k,wi;

	k=i;
	for(j=0;j<AR;j++)
	{
		remove_xmer(k);
		for(wi=0;wi<width;wi++)
			k=tn[k];
	}
	k=i;
	for(j=0;j<AR;j++)
	{
		deposit_ver_2(k);
		for(wi=0;wi<width;wi++)
			k=rn[k];
	}
}

void hortover_13(int i)
{
	int j,k,wi;

	k=i;
	for(j=0;j<AR;j++)
	{
		remove_xmer(k);
		for(wi=0;wi<width;wi++)
			k=tn[k];
	}
	k=i;
	for(j=0;j<AR;j++)
	{
		deposit_ver_3(k);
		for(wi=0;wi<width;wi++)
			k=rn[k];
	}
}

void vertohor_11(int i)
{
	int j,k,wi;

	k=i;
	for(j=0;j<AR;j++)
	{
		remove_ymer(k);
		for(wi=0;wi<width;wi++)
			k=rn[k];
	}
	k=i;
	for(j=0;j<AR;j++)
	{
		deposit_hor_1(k);
		for(wi=0;wi<width;wi++)
			k=tn[k];
	}
}

void vertohor_12(int i)
{
	int j,k,wi;

	k=i;
	for(j=0;j<AR;j++)
	{
		remove_ymer(k);
		for(wi=0;wi<width;wi++)
			k=rn[k];
	}
	k=i;
	for(j=0;j<AR;j++)
	{
		deposit_hor_2(k);
		for(wi=0;wi<width;wi++)
			k=tn[k];
	}
}

void vertohor_13(int i)
{
	int j,k,wi;

	k=i;
	for(j=0;j<AR;j++)
	{
		remove_ymer(k);
		for(wi=0;wi<width;wi++)
			k=rn[k];
	}
	k=i;
	for(j=0;j<AR;j++)
	{
		deposit_hor_3(k);
		for(wi=0;wi<width;wi++)
			k=tn[k];
	}
}

int hflip_compatibility(int st,int hi)
{
	int i,j;

	for(j=0;j<width;j++)
		st=tn[st];	
	for(i=1;i<AR;i++)
	{
		if(lat[st] == hi)
		{
			for(j=0;j<width;j++)
				st=tn[st];
		}
		else
			return 0;
	}	
	return 1;
}

int vflip_compatibility(int st,int hi)
{
	int i,j;

	for(j=0;j<width;j++)
		st=rn[st];	
	for(i=1;i<AR;i++)
	{
		if(lat[st] == hi)
		{
			for(j=0;j<width;j++)
				st=rn[st];
		}
		else
			return 0;
	}	
	return 1;
}

void flip(int input,int ran)
{
	int site,head;

	site = (ran/LBS)*K*L+(ran % LBS)*K+(input/K)*L+(input % K);
	head=lat[site];
	if(head >= 0)
		return;
	else if(head > -4 )
	{
		if(hflip_compatibility(site,head) == 1)
		{
			if(gsl_rng_uniform(r[taskid])<0.5)	
			{
				if((site % L) % sub == 0)
					hortover_11(site);
				else if((site % L) % sub == 1)
					hortover_12(site);
				else 
					hortover_13(site);
			}
			return;
		}
		/*if(hflip_compatibility(site,-2) == 1)
		{
			if(gsl_rng_uniform(r[taskid])<0.5)	
			{
				if((site % L) % sub == 0)
					hortover_11(site);
				else 
					hortover_12(site);
			}
			return;
		}*/
	}
	else
	{
		if(vflip_compatibility(site,head) == 1)
		{
			if(gsl_rng_uniform(r[taskid])<0.5)	
			{
				if((site / L)%sub == 0)
					vertohor_11(site);
				else if((site / L)%sub == 1)
					vertohor_12(site);
				else
					vertohor_13(site);
			}
			return;
		}
		/*if(vflip_compatibility(site,-4) == 1)
		{
			if(gsl_rng_uniform(r[taskid])<0.5)	
			{
				if((site / L)%sub == 0)
					vertohor_11(site);
				else
					vertohor_12(site);
			}
			return;
		}*/
	}
}

void evolve()
{
	int row,col,fi,gi;

	remove_hor_1();
	#pragma omp for private(row)  
	for(row=0;row<L;row=row+width)
		fill_row_1(row);

	remove_hor_2();
	#pragma omp for private(row) 
	for(row=1;row<L;row=row+width)
		fill_row_2(row);
	
	remove_hor_3();
	#pragma omp for private(row) 
	for(row=2;row<L;row=row+width)
		fill_row_3(row);

	remove_ver_1();
	#pragma omp for private(col)
	for(col=0;col<L;col=col+width)
		fill_col_1(col);

	remove_ver_2();
	#pragma omp for private(col)
	for(col=1;col<L;col=col+width)
		fill_col_2(col);

	remove_ver_3();
	#pragma omp for private(col)
	for(col=2;col<L;col=col+width)
		fill_col_3(col);
	
	/*#pragma omp for private(fi)	
	for(fi=0;fi<nstates;fi++)
	{
		for(gi=0;gi<nturns;gi++)
			flip(fi);
	}*/
}

void take_reading(int j)
{
	int i,qy,qi,qj;
	double tmp_qrx,tmp_qry,tmpr,tmp_r,tmp_c,tmp_qcx,tmp_qcy,tmp,tmpnm;
	
	tmpr=0.0;
	for(qi=0;qi<sub;qi++)
	{
		for(qj=0;qj<sub;qj++)
			tmpr=tmpr+1.0*h[qi][qj];
	}
	tmpr=1.0*width*K*tmpr/(1.0*N);
	tmp_qrx=0.0; tmp_qry=0.0; tmp_qcx=0.0; tmp_qcy=0.0;
	for(qi=0;qi<sub;qi++)
	{
		tmp_qrx=tmp_qrx+h0[qi]*cos(2.0*pi*qi/sub);
		tmp_qry=tmp_qry+h0[qi]*sin(2.0*pi*qi/sub);
		tmp_qcx=tmp_qcx+h0[qi+sub]*cos(2.0*pi*qi/sub);
		tmp_qcy=tmp_qcy+h0[qi+sub]*sin(2.0*pi*qi/sub);
	}
	tmp_r=1.0*width*K*pow((tmp_qrx*tmp_qrx+tmp_qry*tmp_qry),0.5)/(1.0*N);
	tmp_c=1.0*width*K*pow((tmp_qcx*tmp_qcx+tmp_qcy*tmp_qcy),0.5)/(1.0*N);
	tmp=fabs(tmp_r-tmp_c); 
	tmpnm=1.0*width*K*(noh-nov)/(1.0*N);

	meanrho[j]=meanrho[j]+tmpr;
	rho2[j]=rho2[j]+tmpr*tmpr;
	meanm1[j]=meanm1[j]+tmp; 
	meanm2[j]=meanm2[j]+tmp*tmp;
	meanm4[j]=meanm4[j]+tmp*tmp*tmp*tmp;
	qnem1[j]=qnem1[j]+fabs(tmpnm);
	qnem2[j]=qnem2[j]+tmpnm*tmpnm;
	qnem4[j]=qnem4[j]+tmpnm*tmpnm*tmpnm*tmpnm;

	qy=floor(BINSIZEQ*tmp);
	binq[qy]=binq[qy]+1.0;

	i=floor((tmpnm+1.0)/BINSIZE);
	if((noh-nov)*width*K==-N)
		i=0;
	if((noh-nov)*width*K==N)
		i=BINS-1;
	if((i<BINS) && (i>=0))
		count[i]++;

	i=floor(tmpr/BINSIZER);
	if((noh+nov)*width*K==N)
		i=BINS-1;
	if((i<BINS) && (i>=0))
		countr[i]++;
	
}

void output2(int t)
{
	double tmp,tmp0,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7,tmp8,tmp9,tmp10,tmp_qrx,tmp_qry,tmp_qcx,tmp_qcy,tmp_r,tmp_c,tmp_qx,tmp_qy,tmpr;
	int qi,qj;
	FILE *fp;

	tmp_qrx=0.0; tmp_qry=0.0; tmp_qcx=0.0; tmp_qcy=0.0;
	tmpr=0.0;
	for(qi=0;qi<sub;qi++)
	{
		for(qj=0;qj<sub;qj++)
			tmpr=tmpr+1.0*h[qi][qj];
	}
	tmpr=1.0*width*K*tmpr/(1.0*N);
	for(qi=0;qi<sub;qi++)
	{
		tmp_qrx=tmp_qrx+h0[qi]*cos(2.0*pi*qi/sub);
		tmp_qry=tmp_qry+h0[qi]*sin(2.0*pi*qi/sub);
		tmp_qcx=tmp_qcx+h0[qi+sub]*cos(2.0*pi*qi/sub);
		tmp_qcy=tmp_qcy+h0[qi+sub]*sin(2.0*pi*qi/sub);
	}
	tmp_r=1.0*width*K*pow((tmp_qrx*tmp_qrx+tmp_qry*tmp_qry),0.5)/(1.0*N);
	tmp_c=1.0*width*K*pow((tmp_qcx*tmp_qcx+tmp_qcy*tmp_qcy),0.5)/(1.0*N);
	tmp=fabs(tmp_r-tmp_c);

	sum_tmpq=sum_tmpq+tmp; sum_tmpr=sum_tmpr+tmpr;
	num_tmp=num_tmp+1.0;
	/*tmp_qx=0.0; tmp_qy=0.0;
	tmp_qx=h0[0]+h0[3]*cos(2.0*pi/6.0)+h0[1]*cos(4.0*pi/6.0)+h0[4]*cos(6.0*pi/6.0)+h0[2]*cos(8.0*pi/6.0)+h0[5]*cos(10*pi/6.0);
	tmp_qy=h0[3]*sin(2.0*pi/6.0)+h0[1]*sin(4.0*pi/6.0)+h0[4]*sin(6.0*pi/6.0)+h0[2]*sin(8.0*pi/6.0)+h0[5]*sin(10*pi/6.0);
	tmp0=1.0*width*K*pow((tmp_qx*tmp_qx+tmp_qy*tmp_qy),0.5)/(1.0*N);*/
	/*tmp2=1.0*width*K*h[0][0]/(1.0*N); tmp3=1.0*width*K*h[0][1]/(1.0*N); tmp4=1.0*width*K*h[0][2]/(1.0*N); 
	tmp5=1.0*width*K*h[1][0]/(1.0*N); tmp6=1.0*width*K*h[1][1]/(1.0*N); tmp7=1.0*width*K*h[1][2]/(1.0*N);
	tmp8=1.0*width*K*h[2][0]/(1.0*N); tmp9=1.0*width*K*h[2][1]/(1.0*N); tmp10=1.0*width*K*h[2][2]/(1.0*N);
	tmp1=1.0*width*K*(noh-nov)/(1.0*N);*/

	if(t % GAP2 == 0)
	{
		tmp2=1.0*width*K*h[0][0]/(1.0*N); tmp3=1.0*width*K*h[0][1]/(1.0*N); tmp4=1.0*width*K*h[0][2]/(1.0*N); 
		tmp5=1.0*width*K*h[1][0]/(1.0*N); tmp6=1.0*width*K*h[1][1]/(1.0*N); tmp7=1.0*width*K*h[1][2]/(1.0*N);
		tmp8=1.0*width*K*h[2][0]/(1.0*N); tmp9=1.0*width*K*h[2][1]/(1.0*N); tmp10=1.0*width*K*h[2][2]/(1.0*N);
		tmp1=1.0*width*K*(noh-nov)/(1.0*N);
		fp=fopen(outfile2,"a");
		fprintf(fp,"%d\t%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\n",t,sum_tmpr/num_tmp,sum_tmpq/num_tmp,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7,tmp8,tmp9,tmp10);
		fclose(fp);
		sum_tmpq=0.0; sum_tmpr=0.0; num_tmp=0.0;
	}
}

void output1(int ave,int window)
{
	int i,p;
	double sum1,sum2,tmp,tmp1,tmp2,avgrho;
	FILE *fp,*fp_,*fpn;

	tmp=1.0/(1.0*ave);
	tmp1=1.0/(window*1.0);
	tmp2=sqrt(tmp1);
	for(i=0;i<window;i++)
	{
		Rho[i]=meanrho[i]*tmp;
		Rho_2[i]=rho2[i]*tmp;
		
	}
	for(i=0;i<window;i++)
	{		
		Q_1[i]=meanm1[i]*tmp/Rho[i];  
		Q_2[i]=meanm2[i]*tmp/pow(Rho[i],2.0); 
		Q_4[i]=meanm4[i]*tmp/pow(Rho[i],4.0);
		QN1[i]=qnem1[i]*tmp/Rho[i];  
		QN2[i]=qnem2[i]*tmp/pow(Rho[i],2.0); 
		QN4[i]=qnem4[i]*tmp/pow(Rho[i],4.0);
	}
	
	for(i=0;i<window;i++)
	{
		fluc_r[i]=(Rho_2[i]-Rho[i]*Rho[i])*L*L; 
		fluc_Q[i]=(Q_2[i]-Q_1[i]*Q_1[i])*L*L;
		fluc_QN[i]=(QN2[i]-QN1[i]*QN1[i])*L*L;
	}

	fp=fopen(outfile1,"w");
	fprintf(fp,"%e	%e",1.0*window*ave,mu);

	sum1=0;sum2=0;
	for(i=0;i<window;i++)
	{
		sum1=sum1+Rho[i];
		sum2=sum2+Rho[i]*Rho[i];
	}
	sum1=sum1*tmp1;sum2=sum2*tmp1;
	fprintf(fp," %13.12e %e",sum1,sqrt(sum2-sum1*sum1)*tmp2);
	
	sum1=0;sum2=0;
	for(i=0;i<window;i++)
	{
		sum1=sum1+Rho_2[i];
		sum2=sum2+Rho_2[i]*Rho_2[i];
	}
	sum1=sum1*tmp1;sum2=sum2*tmp1;
	fprintf(fp," %13.12e %e",sum1,sqrt(sum2-sum1*sum1)*tmp2);

	sum1=0;sum2=0;
	for(i=0;i<window;i++)
	{
		sum1=sum1+Q_1[i];
		sum2=sum2+Q_1[i]*Q_1[i];
	}
	sum1=sum1*tmp1;sum2=sum2*tmp1;
	fprintf(fp," %13.12e %e",sum1,sqrt(sum2-sum1*sum1)*tmp2);

	sum1=0;sum2=0;
	for(i=0;i<window;i++)
	{
		sum1=sum1+Q_2[i];
		sum2=sum2+Q_2[i]*Q_2[i];
	}
	sum1=sum1*tmp1;sum2=sum2*tmp1;
	fprintf(fp," %13.12e %e",sum1,sqrt(sum2-sum1*sum1)*tmp2);

	sum1=0;sum2=0;
	for(i=0;i<window;i++)
	{
		sum1=sum1+Q_4[i];
		sum2=sum2+Q_4[i]*Q_4[i];
	}
	sum1=sum1*tmp1;sum2=sum2*tmp1;
	fprintf(fp," %13.12e %e",sum1,sqrt(sum2-sum1*sum1)*tmp2);

	sum1=0;sum2=0;
	for(i=0;i<window;i++)
	{
		sum1=sum1+QN1[i];
		sum2=sum2+QN1[i]*QN1[i];
	}
	sum1=sum1*tmp1;sum2=sum2*tmp1;
	fprintf(fp," %13.12e %e",sum1,sqrt(sum2-sum1*sum1)*tmp2);

	sum1=0;sum2=0;
	for(i=0;i<window;i++)
	{
		sum1=sum1+QN2[i];
		sum2=sum2+QN2[i]*QN2[i];
	}
	sum1=sum1*tmp1;sum2=sum2*tmp1;
	fprintf(fp," %13.12e %e",sum1,sqrt(sum2-sum1*sum1)*tmp2);

	sum1=0;sum2=0;
	for(i=0;i<window;i++)
	{
		sum1=sum1+QN4[i];
		sum2=sum2+QN4[i]*QN4[i];
	}
	sum1=sum1*tmp1;sum2=sum2*tmp1;
	fprintf(fp," %13.12e %e",sum1,sqrt(sum2-sum1*sum1)*tmp2);	

	sum1=0;sum2=0;
	for(i=0;i<window;i++)
	{
		sum1=sum1+fluc_r[i];
		sum2=sum2+fluc_r[i]*fluc_r[i];
	}
	sum1=sum1*tmp1;sum2=sum2*tmp1;
	fprintf(fp," %13.12e %e",sum1,sqrt(sum2-sum1*sum1)*tmp2);

	sum1=0;sum2=0;
	for(i=0;i<window;i++)
	{
		sum1=sum1+fluc_Q[i];
		sum2=sum2+fluc_Q[i]*fluc_Q[i];
	}
	sum1=sum1*tmp1;sum2=sum2*tmp1;
	fprintf(fp," %13.12e %e",sum1,sqrt(sum2-sum1*sum1)*tmp2);

	sum1=0;sum2=0;
	for(i=0;i<window;i++)
	{
		sum1=sum1+fluc_QN[i];
		sum2=sum2+fluc_QN[i]*fluc_QN[i];
	}
	sum1=sum1*tmp1;sum2=sum2*tmp1;
	fprintf(fp," %13.12e %e\n",sum1,sqrt(sum2-sum1*sum1)*tmp2);

	fclose(fp);
	

	fp=fopen(outfile3,"w");
	fp_=fopen(outfile4,"w");
	fpn=fopen(outfile5,"w");
	fprintf(fp,"#rho P(rho)\n");
	fprintf(fp_,"#rho P(q)\n");
	fprintf(fpn,"#rho P(qnem)\n");

	for(i=0;i<BINSQ;i++)
		fprintf(fp_,"%e\t%e\n",1.0*i/BINSIZEQ,binq[i]/ave/window);

	for(i=0;i<BINS;i++)
	{
		if(count[i]!=0)
			fprintf(fpn,"%e\t%e\n",mass[i],count[i]/factor[i]/ave/window);
	}
	
	for(i=0;i<BINS;i++)
	{
		if(countr[i]!=0)
			fprintf(fp,"%e %e\n",massr[i],countr[i]/factorr[i]/ave/window);
	}	

	fclose(fp); fclose(fp_); fclose(fpn);

}

void measure()
{
	int ind,xtmp,ytmp,qi,qj;
	nov=0; noh=0;
	for(qi=0;qi<sub;qi++)
	{
		for(qj=0;qj<sub;qj++)
			h[qi][qj]=0.0;
	}
	
	for(ind=0;ind<N;ind++)
	{
		
		if(lat[ind] < 0)
		{		
			xtmp=(ind%L)%sub; ytmp=(ind/L)%sub;
			h[ytmp][xtmp]=h[ytmp][xtmp]+1;
			if(lat[ind] < -3)
				nov++;
			else
				noh++;
		}	
	}
	h0[0]=h[0][0]+h[0][1]+h[0][2]; h0[1]=h[1][0]+h[1][1]+h[1][2]; h0[2]=h[2][0]+h[2][1]+h[2][2];
	h0[3]=h[0][0]+h[1][0]+h[2][0]; h0[4]=h[0][1]+h[1][1]+h[2][1]; h0[5]=h[0][2]+h[1][2]+h[2][2];

}

void lat_initialization()
{
	int i;

	if(INITIAL_FLAG == 0)
	{
		for(i=0;i<N;i++)
			lat[i]=0;
	}
	else if(INITIAL_FLAG == 1)
	{
		for(i=0;i<N/2;i++)
			lat[i]=1;
		for(i=N/2;i<N;i++)
                        lat[i]=4;
	}
	sum_tmpq=0.0; sum_tmpr=0.0; num_tmp=0.0;
}

int main (int argc, char *argv[])
{
	int ms,j,g,k,l,p,fi; 
          
	gsl_rng_env_setup();
	T = gsl_rng_default;
	
	//seedval[nthreads]=9103774;
	scanf("%ld",&seedval1);
	seedval[nthreads]=seedval1;
	r[nthreads] = gsl_rng_alloc (T);
	gsl_rng_set(r[nthreads],seedval[nthreads]);
	for(ms=0;ms<nthreads;ms++)
	{		
		r[ms] = gsl_rng_alloc (T);
		seedval[ms] = gsl_rng_uniform_int(r[nthreads],100000000);	
		gsl_rng_set(r[ms],seedval[ms]);
	}

	neighbour();
	//mu=5.45;
	scanf("%lf",&mu);
	initialize();
	lat_init();
	lat_initialization();

	for(ms=0;ms<T_eq;ms++)
	{
		#pragma omp parallel shared(lat)
		{
			taskid = omp_get_thread_num();
			evolve();
		}

		for(p=0;p<nsize;p++)
		{		
			l = floor(gsl_rng_uniform(r[0])*nsize);
			#pragma omp parallel for private(fi)	
			for(fi=0;fi<nturns;fi++)
				flip(l,fi);
		}	
		
		measure();	
		output2(ms);

	}

	for(j=0;j<BLOCKS;j++)
	{
		for(ms=0;ms<AVE;ms++)
		{
			#pragma omp parallel shared(lat)
			{
				taskid = omp_get_thread_num();
				evolve();	
			}

			for(p=0;p<nsize;p++)
			{		
				l = floor(gsl_rng_uniform(r[0])*nsize);
				#pragma omp parallel for private(fi)	
				for(fi=0;fi<nturns;fi++)
					flip(l,fi);
			}	

			measure();				
			output2(T_eq+j*AVE+ms);
			take_reading(j);
		}
		g=j+1;
		output1(AVE,g);
	}

	for(ms=0;ms<=nthreads;ms++)
		gsl_rng_free (r[ms]);		
}
