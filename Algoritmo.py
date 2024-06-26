import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

# Función de aptitud
def funcion_aptitud(x):
    return x**2  # Función de aptitud

# Crear individuo
def crear_individuo(longitud):
    return np.random.randint(2, size=longitud)

# Cadena de bits a valor real
def decodificar_individuo(individuo):
    cadena_binaria = ''.join(str(bit) for bit in individuo)
    return int(cadena_binaria, 2)

# Calcular aptitud
def calcular_aptitud(individuo):
    x = decodificar_individuo(individuo)
    return funcion_aptitud(x)

# Selección
def seleccion(poblacion, aptitudes):
    probabilidades = aptitudes / np.sum(aptitudes) 
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

# Aplicar estrategia de poda para individuos repetidos
def poda_individuos_repetidos(poblacion, aptitudes):
    unique_indices = np.unique(poblacion, axis=0, return_index=True)[1]
    poblacion_unica = [poblacion[i] for i in unique_indices]
    aptitudes_unicas = [aptitudes[i] for i in unique_indices]
    return poblacion_unica, aptitudes_unicas

# Algoritmo genético
def algoritmo_genetico(tam_poblacion, longitud, generaciones, tasa_cruce, tasa_mutacion_generacion, tasa_mutacion_individuo):
    poblacion = [crear_individuo(longitud) for _ in range(tam_poblacion)]
    historia = []

    for generacion in range(generaciones):
        aptitudes = np.array([calcular_aptitud(ind) for ind in poblacion])

        # Aplicar estrategia de poda para individuos repetidos
        poblacion, aptitudes = poda_individuos_repetidos(poblacion, aptitudes)

        mejor_aptitud = np.max(aptitudes)
        peor_aptitud = np.min(aptitudes)
        promedio_aptitud = np.mean(aptitudes)
        mejor_individuo = poblacion[np.argmax(aptitudes)]
        historia.append((poblacion.copy(), mejor_aptitud, peor_aptitud, promedio_aptitud))

        poblacion_seleccionada = seleccion(poblacion, aptitudes)
        nueva_poblacion = []
        for i in range(0, len(poblacion_seleccionada), 2): 
            if i + 1 < len(poblacion_seleccionada):  
                padre1, padre2 = poblacion_seleccionada[i], poblacion_seleccionada[i+1]
                hijo1 = mutacion(cruce(padre1, padre2, tasa_cruce), tasa_mutacion_individuo)
                hijo2 = mutacion(cruce(padre2, padre1, tasa_cruce), tasa_mutacion_individuo)
                nueva_poblacion.extend([hijo1, hijo2])

        poblacion = nueva_poblacion[:tam_poblacion]

        # Aplicar mutación por generación
        if np.random.rand() < tasa_mutacion_generacion:
            for ind in poblacion:
                ind = mutacion(ind, tasa_mutacion_individuo)

    return historia

# Generar video
def generar_video(historia):
    generaciones = len(historia)
    mejores_aptitudes = [gen_data[1] for gen_data in historia]
    peores_aptitudes = [gen_data[2] for gen_data in historia]
    promedios_aptitudes = [gen_data[3] for gen_data in historia]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    out = cv2.VideoWriter('evolucion_aptitud.mp4', fourcc, 1, (800, 600))

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
        
        # Guardar el gráfico
        temp_file = f"temp_plot_{i}.png"
        plt.savefig(temp_file)
        plt.close()
        
        # Leer la imagen
        img = cv2.imread(temp_file)
        out.write(img)
        
        # Eliminar el archivo temporal
        import os
        os.remove(temp_file)

    out.release()

# Ejecutar el algoritmo
def ejecutar_algoritmo():
    try:
        tam_poblacion = int(tam_poblacion_var.get())
        longitud = int(longitud_var.get())
        generaciones = int(generaciones_var.get())
        tasa_cruce = float(tasa_cruce_var.get())
        tasa_mutacion_generacion = float(tasa_mutacion_generacion_var.get())
        tasa_mutacion_individuo = float(tasa_mutacion_individuo_var.get())

        # Ejecutar el algoritmo
        historia = algoritmo_genetico(tam_poblacion, longitud, generaciones, tasa_cruce, tasa_mutacion_generacion, tasa_mutacion_individuo)

        # Generar video de la evolución
        generar_video(historia)

        # Mostrar tabla con los datos de la mejor solución
        ventana_tabla = tk.Toplevel()
        ventana_tabla.title("Mejor Solución ")

        poblacion_final = historia[-1][0]
        aptitudes_finales = [calcular_aptitud(ind) for ind in poblacion_final]
                # Obtener mejor individuo
        mejor_individuo = poblacion_final[np.argmax(aptitudes_finales)]
        indice_mejor = np.argmax(aptitudes_finales)
        mejor_x = decodificar_individuo(mejor_individuo)
        mejor_aptitud = aptitudes_finales[indice_mejor]

        # Crear la tabla
        frame = ttk.Frame(ventana_tabla, padding="10")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        columnas = ["Cadena de Bits", "Índice", "Valor de x", "Valor de Aptitud"]
        tabla = ttk.Treeview(frame, columns=columnas, show='headings')

        for col in columnas:
            tabla.heading(col, text=col)
            tabla.column(col, width=150, anchor=tk.CENTER)

        tabla.insert('', 'end', values=(
            ''.join(map(str, mejor_individuo)),
            indice_mejor,
            mejor_x,
            mejor_aptitud
        ))

        tabla.pack()

    except ValueError:
        messagebox.showerror("Error", "Por favor, ingrese valores válidos.")
        return

# Interfaz gráfica
root = tk.Tk()
root.title("Algoritmo Genético")

frame_principal = ttk.Frame(root, padding="10")
frame_principal.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

ttk.Label(frame_principal, text="Tamaño de la población:").grid(row=0, column=0, sticky=tk.W, pady=2)
ttk.Label(frame_principal, text="Longitud del individuo (bits):").grid(row=1, column=0, sticky=tk.W, pady=2)
ttk.Label(frame_principal, text="Número de generaciones:").grid(row=2, column=0, sticky=tk.W, pady=2)
ttk.Label(frame_principal, text="Tasa de cruce:").grid(row=3, column=0, sticky=tk.W, pady=2)
ttk.Label(frame_principal, text="Tasa de mutación por generación:").grid(row=4, column=0, sticky=tk.W, pady=2)
ttk.Label(frame_principal, text="Tasa de mutación por individuo:").grid(row=5, column=0, sticky=tk.W, pady=2)

tam_poblacion_var = tk.StringVar()
longitud_var = tk.StringVar()
generaciones_var = tk.StringVar()
tasa_cruce_var = tk.StringVar()
tasa_mutacion_generacion_var = tk.StringVar()
tasa_mutacion_individuo_var = tk.StringVar()

ttk.Entry(frame_principal, textvariable=tam_poblacion_var).grid(row=0, column=1, pady=2)
ttk.Entry(frame_principal, textvariable=longitud_var).grid(row=1, column=1, pady=2)
ttk.Entry(frame_principal, textvariable=generaciones_var).grid(row=2, column=1, pady=2)
ttk.Entry(frame_principal, textvariable=tasa_cruce_var).grid(row=3, column=1, pady=2)
ttk.Entry(frame_principal, textvariable=tasa_mutacion_generacion_var).grid(row=4, column=1, pady=2)
ttk.Entry(frame_principal, textvariable=tasa_mutacion_individuo_var).grid(row=5, column=1, pady=2)

ttk.Button(frame_principal, text="Ejecutar Algoritmo", command=ejecutar_algoritmo).grid(row=6, columnspan=2, pady=5)

root.mainloop()

