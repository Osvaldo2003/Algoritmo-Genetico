import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

# Función de aptitud
def funcion_aptitud(x):
    return x**2  # Ejemplo de función de aptitud

# Crear individuo
def crear_individuo(longitud):
    return np.random.randint(2, size=longitud)

# Decodificar cadena de bits a valor real
def decodificar_individuo(individuo):
    cadena_binaria = ''.join(str(bit) for bit in individuo)
    return int(cadena_binaria, 2)

# Calcular aptitud
def calcular_aptitud(individuo):
    x = decodificar_individuo(individuo)
    return funcion_aptitud(x)

# Selección
def seleccion(poblacion, aptitudes):
    probabilidades = aptitudes / aptitudes.sum()
    indices_seleccionados = np.random.choice(len(poblacion), size=len(poblacion), p=probabilidades)
    return np.array(poblacion)[indices_seleccionados]

# Cruce
def cruce(padre1, padre2, tasa_cruce=0.7):
    if np.random.rand() < tasa_cruce:
        punto = np.random.randint(1, len(padre1))
        return np.concatenate([padre1[:punto], padre2[punto:]])
    else:
        return padre1

# Mutación
def mutacion(individuo, tasa_mutacion=0.01):
    for i in range(len(individuo)):
        if np.random.rand() < tasa_mutacion:
            individuo[i] = 1 - individuo[i]
    return individuo

# Algoritmo genético
def algoritmo_genetico(tam_poblacion, longitud, generaciones, tasa_cruce, tasa_mutacion):
    poblacion = [crear_individuo(longitud) for _ in range(tam_poblacion)]
    historia = []

    for generacion in range(generaciones):
        aptitudes = np.array([calcular_aptitud(ind) for ind in poblacion])
        
        mejor_aptitud = np.max(aptitudes)
        peor_aptitud = np.min(aptitudes)
        promedio_aptitud = np.mean(aptitudes)
        mejor_individuo = poblacion[np.argmax(aptitudes)]
        historia.append((poblacion.copy(), mejor_aptitud, peor_aptitud, promedio_aptitud))
        
        poblacion_seleccionada = seleccion(poblacion, aptitudes)
        nueva_poblacion = []
        for i in range(0, tam_poblacion, 2):
            padre1, padre2 = poblacion_seleccionada[i], poblacion_seleccionada[i+1]
            hijo1 = mutacion(cruce(padre1, padre2, tasa_cruce), tasa_mutacion)
            hijo2 = mutacion(cruce(padre2, padre1, tasa_cruce), tasa_mutacion)
            nueva_poblacion.extend([hijo1, hijo2])
        
        poblacion = nueva_poblacion[:tam_poblacion]

    return historia

# Generar video de la evolución de las gráficas
def generar_video(historia):
    generaciones = len(historia)
    mejores_aptitudes = [gen_data[1] for gen_data in historia]
    peores_aptitudes = [gen_data[2] for gen_data in historia]
    promedios_aptitudes = [gen_data[3] for gen_data in historia]

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('evolucion_aptitud.avi', fourcc, 1, (800, 600))

    for i in range(generaciones):
        plt.figure(figsize=(8, 6))
        plt.plot(mejores_aptitudes[:i+1], label='Mejor Aptitud', color='green')
        plt.plot(peores_aptitudes[:i+1], label='Peor Aptitud', color='red')
        plt.plot(promedios_aptitudes[:i+1], label='Aptitud Promedio', color='blue')
        plt.xlabel('Generaciones')
        plt.ylabel('Aptitud')
        plt.legend()
        plt.title('Evolución de la Aptitud de la Población')
        plt.grid(True)
        
        # Guardar el gráfico en un archivo temporal
        temp_file = f"temp_plot_{i}.png"
        plt.savefig(temp_file)
        plt.close()
        
        # Leer la imagen del archivo temporal y escribirla en el video
        img = cv2.imread(temp_file)
        out.write(img)
        
        # Eliminar el archivo temporal
        import os
        os.remove(temp_file)

    out.release()

# Ejecutar el algoritmo y mostrar resultados
def ejecutar_algoritmo():
    try:
        tam_poblacion = int(tam_poblacion_var.get())
        longitud = int(longitud_var.get())
        generaciones = int(generaciones_var.get())
        tasa_cruce = float(tasa_cruce_var.get())
        tasa_mutacion = float(tasa_mutacion_var.get())
    except ValueError:
        messagebox.showerror("Error", "Por favor, ingrese valores válidos.")
        return

    # Ejecutar el algoritmo
    historia = algoritmo_genetico(tam_poblacion, longitud, generaciones, tasa_cruce, tasa_mutacion)

    # Generar el video de la evolución de las gráficas
    generar_video(historia)

    # Crear y mostrar la ventana de la tabla con los datos de la mejor solución
    ventana_tabla = tk.Toplevel()
    ventana_tabla.title("Mejor Solución Encontrada")

    poblacion_final = historia[-1][0]
    aptitudes_finales = [calcular_aptitud(ind) for ind in poblacion_final]
    mejor_individuo = poblacion_final[np.argmax(aptitudes_finales)]
    indice_mejor = np.argmax(aptitudes_finales)
    mejor_x = decodificar_individuo(mejor_individuo)
    mejor_aptitud = aptitudes_finales[indice_mejor]

    datos_tabla = {
        'Cadena de Bits': [''.join(map(str, mejor_individuo))],
        'Índice': [indice_mejor],
        'Valor de x': [mejor_x],
        'Valor de Aptitud': [mejor_aptitud]
    }
    df = pd.DataFrame(datos_tabla)
    
    tabla = ttk.Treeview(ventana_tabla)
    tabla['columns'] = list(datos_tabla.keys())
    tabla.heading("#0", text="Item")
    for col in datos_tabla.keys():
        tabla.heading(col, text=col)
        tabla.column(col, width=100)
    for i, (clave, valor) in enumerate(datos_tabla.items()):
        tabla.insert('', 'end', text=clave, values=valor)

    tabla.pack()

# Crear la interfaz gráfica para ingresar los parámetros del algoritmo
root = tk.Tk()
root.title("Algoritmo Genético")

tk.Label(root, text="Tamaño de la población:").grid(row=0, column=0)
tk.Label(root, text="Longitud del individuo (bits):").grid(row=1, column=0)
tk.Label(root, text="Número de generaciones:").grid(row=2, column=0)
tk.Label(root, text="Tasa de cruce:").grid(row=3, column=0)
tk.Label(root, text="Tasa de mutación:").grid(row=4, column=0)

tam_poblacion_var = tk.StringVar()
longitud_var = tk.StringVar()
generaciones_var = tk.StringVar()
tasa_cruce_var = tk.StringVar()
tasa_mutacion_var = tk.StringVar()

tk.Entry(root, textvariable=tam_poblacion_var).grid(row=0, column=1)
tk.Entry(root, textvariable=longitud_var).grid(row=1, column=1)
tk.Entry(root, textvariable=generaciones_var).grid(row=2, column=1)
tk.Entry(root, textvariable=tasa_cruce_var).grid(row=3, column=1)
tk.Entry(root, textvariable=tasa_mutacion_var).grid(row=4, column=1)

tk.Button(root, text="Ejecutar Algoritmo", command=ejecutar_algoritmo).grid(row=5, columnspan=2)

# Ejecutar la ventana principal
root.mainloop()
