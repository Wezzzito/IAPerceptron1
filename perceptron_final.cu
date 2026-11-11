#include <iostream>
#include <vector>
#include <fstream>
#include <cstdint>
#include <string>
#include <stdexcept>
#include "cuda_runtime.h"

using namespace std;

const int tam_imagen = 784;
const int entradas = 785; // 784 pixeles + bias
const int grupos = 10;   // 10 salidas, una por digito
const int epochs = 10;

const float lr = 0.1f;

float* W = nullptr;
float* imagenGPU = nullptr;
float* salidasGPU = nullptr;
int* erroresGPU = nullptr;

string rutaArchivos = ".";


void revisar(cudaError_t estado) {
    if (estado != cudaSuccess)
        throw runtime_error(cudaGetErrorString(estado));
}

// 1. PRODUCTO PUNTO DEL PERCEPTRON
// u = w[0]*X[0] + w[1]*X[1] + ... + w[784]*X[784]


__global__ void kernelForward(const float* W, const float* X, float* u) {
    int digito = blockIdx.x;
    int hilo = threadIdx.x;

    __shared__ float productos[entradas];

    for (int j = hilo; j < entradas; j += blockDim.x)
        productos[j] = W[digito * entradas + j] * X[j];

    __syncthreads();

    if (hilo == 0) {
        float suma = 0.0f;

        for (int j = 0; j < entradas; j++)
            suma += productos[j];

        u[digito] = suma;
    }
}

// 2. ACTIVACION ESCALON

__device__ int activacion(float u) {
    if (u > 0.0f)
        return 1;
    else
        return 0;
}

// 3. CALCULAR ERROR
// d = resultado esperado
// y = salida del perceptron
// error = d - y

__global__ void kernelError(const float* u, int etiqueta, int* errores) {
    int digito = threadIdx.x;

    if (digito < grupos) {
        int d = (etiqueta == digito) ? 1 : 0;
        int y = activacion(u[digito]);

        errores[digito] = d - y;
    }
}

// 4. ACTUALIZAR PESOS DEL PERCEPTRON
// w = w + lr * error * X

__global__ void kernelActualizarPesos(float* W, const float* X, const int* errores, float tasa) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < grupos * entradas) {
        int digito = idx / entradas;
        int j = idx % entradas;

        int error = errores[digito];

        if (error != 0)
            W[idx] = W[idx] + tasa * error * X[j];
    }
}

// 5. CREAR Y LIBERAR EL PERCEPTRON

void liberarPerceptron() {
    cudaFree(W);
    cudaFree(imagenGPU);
    cudaFree(salidasGPU);
    cudaFree(erroresGPU);
}

void construirPerceptron() {
    revisar(cudaMalloc(&W, grupos * entradas * sizeof(float)));

    revisar(cudaMalloc(&imagenGPU, entradas * sizeof(float)));

    revisar(cudaMalloc(&salidasGPU, grupos * sizeof(float)));

    revisar(cudaMalloc(&erroresGPU, grupos * sizeof(int)));

    revisar(cudaMemset(W, 0, grupos * entradas * sizeof(float)));
}

// 6. ENTRENAR UNA IMAGEN
// AQUI SI SE MODIFICAN LOS PESOS.

int entrenar(const float* imagenCPU, int etiqueta) {
    revisar(cudaMemcpy(imagenGPU,imagenCPU, entradas * sizeof(float), cudaMemcpyHostToDevice));

    // Paso 1: calcular u.
    kernelForward<<<grupos, 256>>>( W,imagenGPU,salidasGPU);

    revisar(cudaGetLastError());

    // Paso 2: aplicar escalon y calcular error.
    kernelError<<<1, grupos>>>(salidasGPU,etiqueta,erroresGPU);

    revisar(cudaGetLastError());

    // Paso 3: actualizar pesos.
    kernelActualizarPesos<<<(grupos * entradas + 255) / 256, 256 >>>(W,imagenGPU, erroresGPU, lr);

    revisar(cudaGetLastError());
    revisar(cudaDeviceSynchronize());

    int errores[grupos];

    revisar(cudaMemcpy(errores, erroresGPU, grupos * sizeof(int), cudaMemcpyDeviceToHost));

    // Igual que Python: contar errores binarios.
    // Una imagen puede producir mas de un error.
    int cantidad = 0;

    for (int d = 0; d < grupos; d++) {
        if (errores[d] != 0)
            cantidad++;
    }

    return cantidad;
}

// 7. PREDECIR UNA IMAGEN
// AQUI NO SE MODIFICAN LOS PESOS.
// Se escoge el mayor u, SIN aplicar el escalon.


int predecir(const float* imagenCPU) {
    revisar(cudaMemcpy(imagenGPU, imagenCPU, entradas * sizeof(float), cudaMemcpyHostToDevice));

    kernelForward<<<grupos, 256>>>(W, imagenGPU, salidasGPU);

    revisar(cudaGetLastError());
    revisar(cudaDeviceSynchronize());

    float u[grupos];

    revisar(cudaMemcpy(u, salidasGPU, grupos * sizeof(float), cudaMemcpyDeviceToHost));

    // Equivalente a np.argmax(salidas).
    int mejor = 0;

    for (int d = 1; d < grupos; d++) {
        if (u[d] > u[mejor])
            mejor = d;
    }

    return mejor;
}

// ============================================================
// 8. ESTRUCTURA DE UNA IMAGEN
// ============================================================

struct DatoImagen {
    int clase = 0;
    int pixeles[784];
};

// Leer enteros de las cabeceras IDX.
uint32_t leerEntero(ifstream& archivo) {
    unsigned char b[4];

    if (!archivo.read(reinterpret_cast<char*>(b), 4))
        throw runtime_error("Cabecera IDX incompleta.");

    return (uint32_t(b[0]) << 24)
         | (uint32_t(b[1]) << 16)
         | (uint32_t(b[2]) << 8)
         | uint32_t(b[3]);
}

// ============================================================
// 9. CARGAR IMAGENES Y ETIQUETAS MNIST
// ============================================================

void cargarSet(
    const string& rutaImagenes, const string& rutaEtiquetas, vector<DatoImagen>& destino)
{
    ifstream imagenes(rutaImagenes, ios::binary);
    ifstream etiquetas(rutaEtiquetas, ios::binary);

    if (!imagenes || !etiquetas) {
        throw runtime_error(
            "No se pudieron abrir: " + rutaImagenes
            + " / " + rutaEtiquetas
        );
    }

    uint32_t magicImagenes = leerEntero(imagenes);
    uint32_t cantidad = leerEntero(imagenes);
    uint32_t filas = leerEntero(imagenes);
    uint32_t columnas = leerEntero(imagenes);

    uint32_t magicEtiquetas = leerEntero(etiquetas);
    uint32_t cantidadEtiquetas = leerEntero(etiquetas);

    if (magicImagenes != 2051 ||
        magicEtiquetas != 2049 ||
        filas != 28 ||
        columnas != 28 ||
        cantidad == 0 ||
        cantidad != cantidadEtiquetas ||
        cantidad > 1000000) {
        throw runtime_error("Archivos MNIST IDX invalidos.");
    }

    destino.resize(cantidad);

    for (uint32_t i = 0; i < cantidad; i++) {
        unsigned char etiqueta;
        unsigned char pixeles[784];

        if (!etiquetas.read(
                reinterpret_cast<char*>(&etiqueta), 1) ||
            !imagenes.read(
                reinterpret_cast<char*>(pixeles), 784)) {
            throw runtime_error("Archivo MNIST incompleto.");
        }

        if (etiqueta > 9)
            throw runtime_error("Etiqueta invalida.");

        destino[i].clase = etiqueta;

        for (int p = 0; p < 784; p++)
            destino[i].pixeles[p] = pixeles[p];
    }

    cout << "Cargadas: " << cantidad << " imagenes.\n";
}

// 10. SEPARACION DE DATOS: ENTRENAMIENTO Y PRUEBA

void cargarMNIST(
    vector<DatoImagen>& entrenamiento,
    vector<DatoImagen>& prueba
) {
    // ENTRENAMIENTO: datos para aprender.
    cargarSet(rutaArchivos + "/train-images-idx3-ubyte", rutaArchivos + "/train-labels-idx1-ubyte", entrenamiento);

    // PRUEBA: datos que NO participan en el aprendizaje.
    cargarSet(rutaArchivos + "/t10k-images-idx3-ubyte", rutaArchivos + "/t10k-labels-idx1-ubyte", prueba);
}

// 11. NORMALIZAR PIXELES Y AGREGAR BIAS

void norma_img(const int* fuente, float* destino) {
    destino[0] = 1.0f; // x0 para el bias

    for (int i = 0; i < tam_imagen; i++)
        destino[i + 1] = static_cast<float>(fuente[i] / 255.0);
}

// 12. EVALUAR: ACCURACY Y ERRORES POR DIGITO
// NO MODIFICA PESOS.


void evaluar(const vector<DatoImagen>& datos, const string& nombre)
{
    vector<float> buffer(entradas);

    int total[grupos] = {};
    int fallos[grupos] = {};
    int aciertos = 0;

    for (const DatoImagen& d : datos) {
        norma_img(d.pixeles, buffer.data());

        int prediccion = predecir(buffer.data());

        total[d.clase]++;

        if (prediccion == d.clase)
            aciertos++;
        else
            fallos[d.clase]++;
    }

    cout << "\nEVALUACION EN " << nombre << '\n';

    cout << "Accuracy: "
         << double(aciertos) / datos.size()
         << '\n';

    cout << "Digito real | Total | Mal clasificados\n";

    for (int d = 0; d < grupos; d++) {
        cout << d << " | "
             << total[d] << " | "
             << fallos[d] << '\n';
    }
}

// 13. PROGRAMA PRINCIPAL

int main() {
    try {
        // ----------------------------------------------------
        // A. CARGAR DATOS SEPARADOS
        // ----------------------------------------------------

        vector<DatoImagen> setEntrenamiento;
        vector<DatoImagen> setPrueba;

        cargarMNIST(setEntrenamiento, setPrueba);

        construirPerceptron();

        cout << "\nPerceptron de una capa de salida";
        cout << ", sin capas ocultas.\n";
        cout << "10 clasificadores, pesos en cero";
        cout << ", tasa = " << lr << '\n';

        vector<float> buffer(entradas);

        // B. ENTRENAMIENTO
        // Solo utiliza setEntrenamiento.
        // Todas las imagenes, en orden, durante 10 epocas.

        cout << "\n========== ENTRENAMIENTO ==========\n";

        for (int e = 1; e <= epochs; e++)
        {
            int errores = 0;

            for (const DatoImagen& d : setEntrenamiento) {
                norma_img(d.pixeles, buffer.data());

                errores += entrenar(
                    buffer.data(),
                    d.clase
                );
            }

            cout << "Epoca " << e
                 << ": errores = " << errores
                 << endl;
        }

        // C. EVALUACION FINAL
        // Ya no se modifican los pesos.

        evaluar(setEntrenamiento, "ENTRENAMIENTO");
        evaluar(setPrueba, "PRUEBA");

        // D. MOSTRAR PESOS FINALES

        vector<float> pesos(grupos * entradas);

        revisar(cudaMemcpy(pesos.data(), W, pesos.size() * sizeof(float), cudaMemcpyDeviceToHost));

        cout << "\nPesos del clasificador del digito 0:\n";

        for (int j = 0; j < entradas; j++)
            cout << pesos[j] << ' ';

        cout << "\nPesos bias de los 10 digitos:\n";

        for (int d = 0; d < grupos; d++)
            cout << pesos[d * entradas] << ' ';

        cout << '\n';
        liberarPerceptron();

        return 0;
    }
    catch (const exception& e)
    {
        cerr << "Error: " << e.what() << '\n';
        liberarPerceptron();
        return 1;
    }
}
