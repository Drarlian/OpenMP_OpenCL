# PROJETO: Transformação de Pontos 3D com OpenMP e OpenCL

---

## 1. Descrição Geral

Este projeto implementa a aplicação simultânea de translação e escala em um conjunto de pontos 3D, utilizando coordenadas homogêneas `(x, y, z, 1)` e uma matriz 4×4.

Foram desenvolvidas duas versões em linguagem C:

- **Versão OpenMP (CPU)**: paralelização do laço de processamento dos pontos utilizando OpenMP.
- **Versão OpenCL (GPU)**: execução do núcleo de cálculo (kernel) na GPU, utilizando OpenCL.

Em ambas as versões, os valores de translação (`tx`, `ty`, `tz`) e escala (`sx`, `sy`, `sz`) são fornecidos pelo usuário, e os pontos são gerados aleatoriamente.

---

## 2. Estrutura do Projeto

Arquivos principais:

- `CMakeLists.txt`
- `openmp.c` &rarr; versão OpenMP
- `opencl.c` &rarr; versão OpenCL

Pastas:

- `external/opencl/include/CL`
  - Contém os headers oficiais do OpenCL obtidos do repositório `KhronosGroup/OpenCL-Headers`.

- `external/opencl/bin`
  - Deve conter uma **cópia local** da DLL de runtime do OpenCL (`OpenCL.dll`), não versionada no repositório.

---

## 3. Pré-Requisitos

- Sistema operacional: **Windows**
- Compilador: **MinGW** (configurado pelo CLion)
- **CMake** (integrado ao CLion)
- **OpenMP** habilitado no compilador (uso de `-fopenmp`)
- Runtime de **OpenCL** instalado no sistema (normalmente via driver da GPU)

Além disso, é necessário:

1. Copiar o arquivo `OpenCL.dll` de:
   `C:\Windows\System32\OpenCL.dll`

2. Colar em:
   `external\opencl\bin\OpenCL.dll`

Essa DLL não está junto ao repositório (está listada no `.gitignore`).

---

## 4. Compiliação e Execução (CLion)

1. Abrir o projeto no **CLion**.
2. O CLion deverá detectar automaticamente o `CMakeLists.txt` e configurar os alvos de build:

   - `A4_OpenMP` (versão OpenMP)
   - `A4_OpenCL` (versão OpenCL)

3. Selecionar o alvo desejado no CLion:

   - Para CPU/OpenMP: `A4_OpenMP`
   - Para GPU/OpenCL: `A4_OpenCL`

4. Executar o build (Build).
5. Executar (Run) o alvo selecionado.

Durante a execução, o programa solicitará:

- `tx`, `ty`, `tz` (translação)
- `sx`, `sy`, `sz` (escala)

---

## 5. Observações

- O número de pontos processados é definido por uma constante no código (`N_POINTS`).
- A versão OpenMP mede o tempo sequencial e o tempo paralelo, permitindo calcular o **speedup**.
- A versão OpenCL mede o tempo do kernel na GPU e o tempo total (transferências + kernel).
- As diferenças de desempenho entre OpenMP e OpenCL dependem tanto do hardware utilizado quanto do custo de transferência de dados entre CPU e GPU.