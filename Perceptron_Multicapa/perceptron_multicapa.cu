#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <stdexcept>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <iomanip>
using namespace std;

/*
Julio Pino Charun
Gianella Rosas Lipa
Rodrigo Salas Ramos
*/

const int entradas = 784, ocultas = 32, salidas = 10;
const int epochs = 20, batch_size = 64, hilos = 256;
float lr = 0.1f;
bool usarPromedio = true;
bool mostrarMatrices = false;

void revisar(cudaError_t e) {
    if (e != cudaSuccess) throw runtime_error(cudaGetErrorString(e));
}

float *W1=nullptr, *b1=nullptr, *W2=nullptr, *b2=nullptr;
float *X=nullptr, *D=nullptr, *A1=nullptr, *A2=nullptr;
float *E1=nullptr, *E2=nullptr, *dW1=nullptr, *dW2=nullptr;
float *db1=nullptr, *db2=nullptr;

__global__ void kernelMultiplicar(const float* A, const float* B,float* C, int m, int k, int n,bool tA, bool tB, float escala) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m*n) return;
    int fila = idx/n, col = idx%n;
    float suma = 0.0f;
    for (int p=0; p<k; p++) {
        float a = tA ? A[p*m+fila] : A[fila*k+p];
        float b = tB ? B[col*k+p] : B[p*n+col];
        suma += a*b;
    }
    C[idx] = escala*suma;
}

void multiplicar(const float* A, const float* B, float* C,int m, int k, int n, bool tA=false, bool tB=false,float escala=1.0f) {
    kernelMultiplicar<<<(m*n+255)/256,256>>>(A,B,C,m,k,n,tA,tB,escala);
    revisar(cudaGetLastError());
}

__global__ void kernelSigmoide(float* A, const float* b, int filas, int cols) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < filas*cols) {
        float z = fminf(50.0f, fmaxf(-50.0f, A[i]+b[i%cols]));
        A[i] = 1.0f/(1.0f+expf(-z));
    }
}

__global__ void kernelErrorSalida(const float* A, const float* D,float* E, int cantidad) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i<cantidad) E[i] = D[i]-A[i];
}

__global__ void kernelDerivada(float* E, const float* A, int cantidad) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i<cantidad) E[i] *= A[i]*(1.0f-A[i]);
}

__global__ void kernelMedia(const float* E, float* db, int filas, int cols, float escala) {
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    if (col<cols) {
        float suma=0.0f;
        for (int fila=0; fila<filas; fila++) suma += E[fila*cols+col];
        db[col]=suma*escala;
    }
}

__global__ void kernelActualizar(float* W, const float* gradiente,int cantidad, float tasa) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i<cantidad) W[i] += tasa*gradiente[i];
}
void reservar(float*& p, int cantidad) {
    revisar(cudaMalloc(&p, cantidad*sizeof(float)));
}
void copiarGPU(float* destino, const vector<float>& origen) {
    revisar(cudaMemcpy(destino,origen.data(),origen.size()*sizeof(float),cudaMemcpyHostToDevice));
}
void construirRed(mt19937& rng) {

    reservar(W1,entradas*ocultas); reservar(b1,ocultas);
    reservar(W2,ocultas*salidas); reservar(b2,salidas);
    reservar(X,batch_size*entradas); reservar(D,batch_size*salidas);
    reservar(A1,batch_size*ocultas); reservar(A2,batch_size*salidas);
    reservar(E1,batch_size*ocultas); reservar(E2,batch_size*salidas);
    reservar(dW1,entradas*ocultas); reservar(dW2,ocultas*salidas);
    reservar(db1,ocultas); reservar(db2,salidas);

    normal_distribution<float> dist1(0.0f,sqrtf(1.0f/entradas));
    normal_distribution<float> dist2(0.0f,sqrtf(1.0f/ocultas));

    vector<float> pesos1(entradas*ocultas), pesos2(ocultas*salidas);

    for (float& v:pesos1) 
        v=dist1(rng);
    for (float& v:pesos2) 
        v=dist2(rng);

    copiarGPU(W1,pesos1); copiarGPU(W2,pesos2);

    revisar(cudaMemset(b1,0,ocultas*sizeof(float)));
    revisar(cudaMemset(b2,0,salidas*sizeof(float)));
}
void liberarRed() {
    for (float* p : {W1,b1,W2,b2,X,D,A1,A2,E1,E2,dW1,dW2,db1,db2})
        if (p) cudaFree(p);
}

void forward(int B) {
    multiplicar(X,W1,A1,B,entradas,ocultas);
    kernelSigmoide<<<(B*ocultas+255)/256,256>>>(A1,b1,B,ocultas);
    revisar(cudaGetLastError());
    multiplicar(A1,W2,A2,B,ocultas,salidas);
    kernelSigmoide<<<(B*salidas+255)/256,256>>>(A2,b2,B,salidas);
    revisar(cudaGetLastError());
}

void imprimirMatriz(const string& nombre, const float* gpu, int filas, int cols) {
    vector<float> cpu(size_t(filas)*cols);
    revisar(cudaMemcpy(cpu.data(), gpu, cpu.size()*sizeof(float), cudaMemcpyDeviceToHost));
    cout << nombre << " (" << filas << " x " << cols << ") [primeras filas/columnas]:\n";
    for (int i=0; i<min(filas,4); i++) {
        for (int j=0; j<min(cols,10); j++) 
            cout << setw(11) << cpu[i*cols+j] << ' ';
        if (cols>10) cout << "...";
        cout << '\n';
    }
    if (filas>4) cout << "...\n";
}

void entrenarLote(int B, bool detalle=false) {
    const float escala = usarPromedio ? 1.0f/B : 1.0f;
    forward(B);

    kernelErrorSalida<<<(B*salidas+255)/256,256>>>(A2,D,E2,B*salidas);
    revisar(cudaGetLastError());

    multiplicar(A1,E2,dW2,ocultas,B,salidas,true,false,escala);
    kernelMedia<<<1,256>>>(E2,db2,B,salidas,escala);

    revisar(cudaGetLastError());
    multiplicar(E2,W2,E1,B,salidas,ocultas,false,true);

    kernelDerivada<<<(B*ocultas+255)/256,256>>>(E1,A1,B*ocultas);
    revisar(cudaGetLastError());

    multiplicar(X,E1,dW1,entradas,B,ocultas,true,false,escala);
    kernelMedia<<<1,256>>>(E1,db1,B,ocultas,escala);

    revisar(cudaGetLastError());

    if (detalle) {
        imprimirMatriz("X",X,B,entradas); imprimirMatriz("D",D,B,salidas);
        imprimirMatriz("A1",A1,B,ocultas); imprimirMatriz("Y = A2",A2,B,salidas);
        imprimirMatriz("E2 = D-Y",E2,B,salidas); imprimirMatriz("E1",E1,B,ocultas);
        imprimirMatriz("W1 antes",W1,entradas,ocultas); imprimirMatriz("W2 antes",W2,ocultas,salidas);
        imprimirMatriz("Direccion de actualizacion W1",dW1,entradas,ocultas);
        imprimirMatriz("Direccion de actualizacion W2",dW2,ocultas,salidas);
    }
    kernelActualizar<<<(entradas*ocultas+255)/256,256>>>(W1,dW1,entradas*ocultas,lr);
    revisar(cudaGetLastError());

    kernelActualizar<<<1,256>>>(b1,db1,ocultas,lr);
    revisar(cudaGetLastError());

    kernelActualizar<<<(ocultas*salidas+255)/256,256>>>(W2,dW2,ocultas*salidas,lr);
    revisar(cudaGetLastError());

    kernelActualizar<<<1,256>>>(b2,db2,salidas,lr);
    revisar(cudaGetLastError());
    revisar(cudaDeviceSynchronize());

    if (detalle) {
        imprimirMatriz("W1 actualizado",W1,entradas,ocultas);
        imprimirMatriz("W2 actualizado",W2,ocultas,salidas);
    }
}

struct Dataset {
    vector<float> pixeles;
    vector<int> etiquetas;
};
uint32_t leerEntero(ifstream& f) {
    unsigned char b[4];
    if (!f.read(reinterpret_cast<char*>(b),4)) throw runtime_error("Cabecera incompleta.");
    return (uint32_t(b[0])<<24)|(uint32_t(b[1])<<16)|(uint32_t(b[2])<<8)|b[3];
}
Dataset cargarSet(const string& imagenes, const string& etiquetas) {
    ifstream fi(imagenes,ios::binary), fe(etiquetas,ios::binary);
    if (!fi || !fe) 
        throw runtime_error("No se pudieron abrir: "+imagenes+" / "+etiquetas);

    uint32_t mi=leerEntero(fi), n=leerEntero(fi);
    uint32_t filas=leerEntero(fi), cols=leerEntero(fi);
    uint32_t me=leerEntero(fe), ne=leerEntero(fe);

    if (mi!=2051 || me!=2049 || filas!=28 || cols!=28 || n!=ne || n==0 || n>1000000)
        throw runtime_error("Archivos IDX invalidos.");

    Dataset datos;
    datos.pixeles.resize(size_t(n)*entradas); 
    datos.etiquetas.resize(n);

    for (uint32_t i=0; i<n; i++) {
        unsigned char pixels[784], etiqueta;
        if (!fi.read(reinterpret_cast<char*>(pixels),784) ||!fe.read(reinterpret_cast<char*>(&etiqueta),1))
            throw runtime_error("Archivo MNIST incompleto.");
        if (etiqueta>=10) 
            throw runtime_error("Etiqueta fuera de 0..9.");
        datos.etiquetas[i]=etiqueta;
        for (int j=0; j<entradas; j++)
            datos.pixeles[size_t(i)*entradas+j]=pixels[j]/255.0f;
    }
    return datos;
}

void prepararLote(const Dataset& datos, const vector<int>& orden, int inicio, int B) {
    vector<float> loteX(B*entradas), loteD(B*salidas,0.0f);
    for (int i=0; i<B; i++) {
        int indice=orden[inicio+i];
        copy_n(datos.pixeles.data()+size_t(indice)*entradas,entradas,loteX.data()+i*entradas);

        loteD[i*salidas+datos.etiquetas[indice]]=1.0f;
    }
    copiarGPU(X,loteX); copiarGPU(D,loteD);
}
void evaluar(const Dataset& datos, const string& nombre, bool detalle) {

    int N=static_cast<int>(datos.etiquetas.size());
    vector<int> orden(N); iota(orden.begin(),orden.end(),0);

    int total[10]={}, fallos[10]={}, aciertos=0;
    double perdida=0.0;

    for (int inicio=0; inicio<N; inicio+=batch_size) {
        int B=min(batch_size,N-inicio);
        prepararLote(datos,orden,inicio,B);

        forward(B);
        revisar(cudaDeviceSynchronize());

        vector<float> pred(B*salidas);
        revisar(cudaMemcpy(pred.data(),A2,pred.size()*sizeof(float),cudaMemcpyDeviceToHost));

        for (int i=0; i<B; i++) {
            int real=datos.etiquetas[inicio+i], mejor=0;
            for (int d=0; d<salidas; d++) {
                float valor=pred[i*salidas+d];
                if (!isfinite(valor)) 
                    throw runtime_error("Salida no finita.");
                if (valor>pred[i*salidas+mejor]) 
                    mejor=d;

                double p=max(1e-7,min(1.0-1e-7,double(valor)));
                perdida -= (d==real) ? log(p) : log(1.0-p);
            }
            total[real]++;
            if (mejor==real) aciertos++; else fallos[real]++;
        }
    }
    cout << nombre << " | BCE por imagen: " << perdida/N<< " | Accuracy: " << double(aciertos)/N<< " | Errores: " << N-aciertos << " / " << N << '\n';
    if (detalle) {
        cout << "Digito real | Total | Aciertos | Errores\n";
        for (int d=0; d<salidas; d++)
            cout << d << " | " << total[d] << " | " << total[d]-fallos[d]<< " | " << fallos[d] << '\n';
    }
}

int main(int argc,char* argv[]) {
    try {
        string ruta=".";
        bool rutaDada=false, tasaDada=false;
        for (int i=1; i<argc; i++) {
            string arg=argv[i];
            if (arg=="--suma") 
                usarPromedio=false;
            else if (arg=="--detalle") 
                mostrarMatrices=true;
            else if (arg=="--lr" && i+1<argc) {
                lr=stof(argv[++i]); 
                tasaDada=true;
                if (!isfinite(lr) || lr<=0) 
                    throw runtime_error("Tasa invalida");
            } else if (!rutaDada && arg.rfind("--",0)!=0) { 
                ruta=arg; rutaDada=true; 
            } else 
                throw runtime_error("Uso: multicapa [carpeta_MNIST] [--suma] [--lr tasa] [--detalle]");
        }
        if (!usarPromedio && !tasaDada) 
            lr /= batch_size;
        revisar(cudaSetDevice(0));
        cudaDeviceProp gpu{}; revisar(cudaGetDeviceProperties(&gpu,0));
        cout << "GPU: " << gpu.name << '\n';

        Dataset entrenamiento=cargarSet(ruta+"/train-images-idx3-ubyte",ruta+"/train-labels-idx1-ubyte");
        Dataset prueba=cargarSet(ruta+"/t10k-images-idx3-ubyte",ruta+"/t10k-labels-idx1-ubyte");

        cout << "Train: " << entrenamiento.etiquetas.size() << " | Test: " << prueba.etiquetas.size() << '\n';
        cout << "Red: 784 -> 32 -> 10. Sigmoide, BCE, pesos aleatorios.\n";
        cout << "Epocas: " << epochs << " | Tasa: " << lr << " | Lote: " << batch_size << " | Modo: " << (usarPromedio ? "promedio" : "suma") << '\n';
        
        mt19937 rng(42); construirRed(rng);
        vector<int> orden(entrenamiento.etiquetas.size());
        iota(orden.begin(),orden.end(),0);
        cout << fixed << setprecision(6);

        cout << "\n========== ENTRENAMIENTO ==========\n";
        for (int e=1; e<=epochs; e++) {
            shuffle(orden.begin(),orden.end(),rng);
            for (int inicio=0; inicio<static_cast<int>(orden.size()); inicio+=batch_size) {
                int B=min(batch_size,static_cast<int>(orden.size())-inicio);
                prepararLote(entrenamiento,orden,inicio,B);
                entrenarLote(B,mostrarMatrices && e==1 && inicio==0);
            }
            cout << "Epoca " << e << " | ";
            evaluar(entrenamiento,"TRAIN",false);
        }
        cout << "\n========== EVALUACION FINAL ==========\n";
        evaluar(entrenamiento,"TRAIN",true);
        evaluar(prueba,"TEST",true);

        vector<float> p1(entradas*ocultas), p2(ocultas*salidas);
        revisar(cudaMemcpy(p1.data(),W1,p1.size()*sizeof(float),cudaMemcpyDeviceToHost));
        revisar(cudaMemcpy(p2.data(),W2,p2.size()*sizeof(float),cudaMemcpyDeviceToHost));
        cout << "Primeros 10 pesos hacia la neurona oculta 0:\n";

        for (int j=0;j<10;j++) 
            cout << p1[j*ocultas] << ' ';
        cout << "\nPesos desde la neurona oculta 0 hacia las 10 salidas:\n";
        for (int d=0;d<salidas;d++) 
            cout << p2[d] << ' ';
        cout << '\n';
        liberarRed(); 

        return 0;
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << '\n';
        liberarRed(); 
        return 1;
    }
}
