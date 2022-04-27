#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include <iostream>
#include <iomanip>
using namespace std;

int main(int argc, char* argv[]){
    
    int ell = atoi(argv[1]);
    double max_f,tmp,optima;
    int max_index[160];
    int base;
    srand(time(NULL));
    char filename[50];
for(int n=0;n<100;n++){
    sprintf(filename,"pnk%d_4_5_%d",ell,n);
    FILE* f = fopen(filename,"w");
    fprintf(f,"%d 4 5\n",ell);
    optima = 0.0;
    for(int i=0;i<ell/5;i++){
        max_f = ((double)rand()/(double)RAND_MAX);
        max_index[i] = 0;
        fprintf(f,"%.9lf ",max_f);
        for(int j=1;j<32;j++){
            tmp = ((double)rand()/(double)RAND_MAX);
            fprintf(f,"%.9lf ",tmp);
            if(tmp>max_f){
                max_f = tmp;
                max_index[i] = j;
            }
        }
        optima = optima + max_f;
    }
    fprintf(f,"\n%.9lf\n",optima);
    for(int i=0;i<ell;i++)
        fprintf(f,"%d ",i);
    fprintf(f,"\n");
    for(int i=0;i<ell/5;i++){
        base = 16;
        //printf("index=%d\n",max_index[i]);
        for(int j=0;j<5;j++){
            fprintf(f,"%d",max_index[i]/base);
            //printf("%d",max_index[i]/base);
            max_index[i] = max_index[i]%base;
            base = base/2;
        }
        fprintf(f," ");
        //printf(" ");
    }
    fprintf(f,"\n");
    fclose(f);
}
    return 0;
}
