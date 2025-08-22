import numpy as np
import pandas as pd
from scipy.optimize import linprog

# Função para exibir a tabela Simplex de forma organizada
def print_simplex_tableau(c, A, b, tableau_name="Tabela Simplex"):
    tableau = np.hstack([A, b.reshape(-1, 1)])
    obj_row = np.hstack([c, [0]]) # Função objetivo com LD = 0
    tableau = np.vstack([tableau, obj_row])

    col_names = [f"x{i+1}" for i in range(A.shape[1])] + ["LD"]
    row_names = [f"Eq{i+1}" for i in range(A.shape[0])] + ["Z"]

    df = pd.DataFrame(tableau, columns=col_names, index=row_names)
    print(f"\n{tableau_name}:")
    print(df)

# Definindo os coeficientes da função objetivo e as restrições
def solve_two_phase_simplex():
    # Função objetivo original
    c = np.array([1, 1, 1, 0, 0, 0, 0]) # Min Z = x1 + x2 + x3

    # Matriz de coeficientes das restrições (A)
    A = np.array([
        [ 2, 1, -1, 1, 0, 0, 0], # 2x1 + x2 - x3 + x4 = 10 (folga)
        [ 1, 1, 2, 0, -1, 1, 0], # x1 + x2 + 2x3 - x5 + x6 = 20 (artificial)
        [ 2, 1, 3, 0, 0, 0, 1] # 2x1 + x2 + 3x3 + x7 = 60 (artificial)
    ])

    # Lado direito (LD) dos sistemas de equações
    b = np.array([10, 20, 60])

    # Função objetivo auxiliar para a Fase 1
    c_aux = np.array([0, 0, 0, 0, 0, 1, 1])

    # Mostrando a tabela inicial para a Fase 1
    print_simplex_tableau(c_aux, A, b, tableau_name="Tabela Inicial (Fase 1)")

    # Fase 1 - Resolver a função objetivo auxiliar
    res_phase1 = linprog(c_aux, A_eq=A, b_eq=b, method='simplex')

    # Se a função auxiliar (Fase 1) não consegue atingir Z = 0, o problema é inviável
    if res_phase1.fun > 1e-5: # Tolera um pequeno erro numérico
        print("Problema inviável. Não foi possível eliminar as variáveis artificiais.")
        return

    # Se o problema é viável, seguimos para a Fase 2
    print("\n--- Fase 1 Concluída ---")
    print(f"Função objetivo auxiliar Z1 = {res_phase1.fun}")
    print(f"Solução da Fase 1: {res_phase1.x}")

    # Fase 2 - Remover as variáveis artificiais e resolver a função objetivo original
    A_phase2 = A[:, :5] # Remover as colunas das variáveis artificiais
    c_phase2 = c[:5] # Função objetivo original sem as variáveis artificiais

    # Mostrando a tabela inicial para a Fase 2
    print_simplex_tableau(c_phase2, A_phase2, b, tableau_name="Tabela Inicial (Fase 2)")

    # Resolver a função objetivo original (Fase 2)
    res_phase2 = linprog(c_phase2, A_eq=A_phase2, b_eq=b, method='simplex')

    # Mostrando o resultado final
    print("\n--- Fase 2 Concluída ---")
    print(f"Função objetivo original Z = {res_phase2.fun}")
    print(f"Solução ótima: {res_phase2.x[:3]}") # Solução para x1, x2, x3

# Executar a função
solve_two_phase_simplex()
