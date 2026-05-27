import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ── Parámetros globales ──────────────────────────────────────────────────────
N = 980          # total de estampas distintas
S = 7            # estampas por sobre
P_SOBRE = 9.50   # Q por sobre
P_CAJA  = 975.00 # Q por caja (104 sobres)
SOBRES_CAJA = 104

np.random.seed(42)
print(f"Parámetros: N={N}, S={S}, precio sobre=Q{P_SOBRE}, precio caja=Q{P_CAJA}")


def simular_album(n=N, s=S, objetivo=1.0, intercambio_k=0, semilla=None):
    """
    Simula la compra de sobres hasta completar `objetivo` fracción del álbum.
    Si intercambio_k > 0, cada `intercambio_k` repetidas se convierte en 1 nueva
    (aproximación del intercambio).
    Devuelve (sobres_comprados, repetidas_totales, coleccion_final).
    """
    if semilla is not None:
        np.random.seed(semilla)
    coleccion = np.zeros(n, dtype=bool)
    repetidas = 0
    sobres = 0
    meta = int(np.ceil(objetivo * n))
    intercambio_pool = 0

    while coleccion.sum() < meta:
        nuevas = np.random.randint(0, n, size=s)
        sobres += 1
        for e in nuevas:
            if coleccion[e]:
                repetidas += 1
                intercambio_pool += 1
                if intercambio_k > 0 and intercambio_pool >= intercambio_k:
                    # convertir k repetidas en 1 nueva aleatoria
                    faltantes = np.where(~coleccion)[0]
                    if len(faltantes) > 0:
                        nueva = np.random.choice(faltantes)
                        coleccion[nueva] = True
                    intercambio_pool = 0
            else:
                coleccion[e] = True

    return sobres, repetidas, coleccion.sum()


def simular_lote(n_sim=3000, **kwargs):
    """Ejecuta n_sim simulaciones y devuelve array de sobres comprados."""
    return np.array([simular_album(**kwargs)[0] for _ in range(n_sim)])

print("Funciones base listas.")


N_SIM = 3000

repetidas_lista = []
sobres_lista    = []

for _ in range(N_SIM):
    s_comp, reps, _ = simular_album()
    sobres_lista.append(s_comp)
    repetidas_lista.append(reps)

repetidas_arr = np.array(repetidas_lista)
sobres_arr    = np.array(sobres_lista)

print(f"Sobres promedio para completar el álbum : {sobres_arr.mean():.1f}  (IC 95%: [{np.percentile(sobres_arr,2.5):.0f}, {np.percentile(sobres_arr,97.5):.0f}])")
print(f"Repetidas promedio al completar         : {repetidas_arr.mean():.1f}")
print(f"Estampas totales compradas (promedio)   : {(sobres_arr*S).mean():.1f}")
print(f"% de estampas que son repetidas (prom)  : {repetidas_arr.mean()/(sobres_arr*S).mean()*100:.1f}%")


fig, axes = plt.subplots(1, 2, figsize=(13, 4))

axes[0].hist(repetidas_arr, bins=50, color='steelblue', edgecolor='white', alpha=0.85)
axes[0].axvline(repetidas_arr.mean(), color='crimson', lw=2, label=f'Media = {repetidas_arr.mean():.0f}')
axes[0].set_title('Distribución de estampas repetidas al completar el álbum')
axes[0].set_xlabel('Número de repetidas'); axes[0].set_ylabel('Frecuencia')
axes[0].legend()

# Curva de llenado promedio (1 simulación detallada)
np.random.seed(0)
coleccion = np.zeros(N, dtype=bool)
distintas_por_sobre = []
for _ in range(int(sobres_arr.mean() * 1.05)):
    nuevas = np.random.randint(0, N, size=S)
    for e in nuevas:
        coleccion[e] = True
    distintas_por_sobre.append(coleccion.sum())

axes[1].plot(distintas_por_sobre, color='seagreen', lw=1.5)
axes[1].axhline(N, color='grey', ls='--', lw=1, label='Álbum completo (980)')
axes[1].set_title('Curva de llenado del álbum (una simulación)')
axes[1].set_xlabel('Sobres comprados'); axes[1].set_ylabel('Estampas distintas')
axes[1].legend()

plt.tight_layout()
plt.savefig('p1_repetidas.png', dpi=120, bbox_inches='tight')
plt.show()
print("Gráfica guardada.")


PRESUPUESTO = 4000.0

# Sobres sueltos
sobres_disponibles_sueltos = int(PRESUPUESTO // P_SOBRE)
# Cajas (y sobres sueltos con el sobrante)
cajas  = int(PRESUPUESTO // P_CAJA)
sobrante = PRESUPUESTO - cajas * P_CAJA
sobres_disponibles_cajas = cajas * SOBRES_CAJA + int(sobrante // P_SOBRE)

print(f"Presupuesto: Q{PRESUPUESTO:.2f}")
print(f"  Sobres sueltos comprables        : {sobres_disponibles_sueltos}")
print(f"  Cajas comprables                 : {cajas}  ({cajas*SOBRES_CAJA} sobres) + {int(sobrante//P_SOBRE)} sueltos = {sobres_disponibles_cajas} sobres")


def prob_exito_presupuesto(max_sobres, n_sim=3000):
    exitos = 0
    for _ in range(n_sim):
        coleccion = set()
        for _ in range(max_sobres):
            coleccion.update(np.random.randint(0, N, size=S).tolist())
            if len(coleccion) == N:
                exitos += 1
                break
    return exitos / n_sim

p_sueltos = prob_exito_presupuesto(sobres_disponibles_sueltos)
p_cajas   = prob_exito_presupuesto(sobres_disponibles_cajas)

print(f"P(completar álbum | solo sobres sueltos, Q{PRESUPUESTO:.0f}) = {p_sueltos:.4f}  ({p_sueltos*100:.2f}%)")
print(f"P(completar álbum | solo cajas,          Q{PRESUPUESTO:.0f}) = {p_cajas:.4f}  ({p_cajas*100:.2f}%)")
print(f"Diferencia: {abs(p_cajas-p_sueltos)*100:.2f} pp a favor de {'cajas' if p_cajas>p_sueltos else 'sueltos'}")


# Curva de probabilidad de éxito vs presupuesto
presupuestos = np.arange(500, 8001, 250)
p_sue, p_caj = [], []

for ppto in presupuestos:
    ms = int(ppto // P_SOBRE)
    mc = int(ppto // P_CAJA) * SOBRES_CAJA + int((ppto % P_CAJA) // P_SOBRE)
    p_sue.append(prob_exito_presupuesto(ms, n_sim=1000))
    p_caj.append(prob_exito_presupuesto(mc, n_sim=1000))

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(presupuestos, p_sue, label='Sobres sueltos', color='steelblue', lw=2)
ax.plot(presupuestos, p_caj, label='Cajas', color='darkorange', lw=2, ls='--')
ax.axvline(PRESUPUESTO, color='grey', ls=':', lw=1.5, label=f'Q{PRESUPUESTO:.0f}')
ax.set_xlabel('Presupuesto (Q)'); ax.set_ylabel('P(completar álbum)')
ax.set_title('Probabilidad de completar el álbum según presupuesto y modalidad de compra')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('p2_presupuesto.png', dpi=120, bbox_inches='tight')
plt.show()


K_valores = [0, 2, 3, 4, 5, 6, 8, 10]
resultados_k = {}

for k in K_valores:
    arr = simular_lote(n_sim=2000, intercambio_k=k)
    resultados_k[k] = arr
    media = arr.mean()
    ahorro_sobres = resultados_k[0].mean() - media
    ahorro_q = ahorro_sobres * P_SOBRE
    print(f"K={k:2d} → sobres promedio: {media:7.1f}  | ahorro vs sin intercambio: {ahorro_sobres:7.1f} sobres = Q{ahorro_q:8.2f}")


medias  = [resultados_k[k].mean()  for k in K_valores]
p25     = [np.percentile(resultados_k[k], 25) for k in K_valores]
p75     = [np.percentile(resultados_k[k], 75) for k in K_valores]

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(K_valores, medias, 'o-', color='steelblue', lw=2, label='Media')
ax.fill_between(K_valores, p25, p75, alpha=0.25, color='steelblue', label='IQR (P25–P75)')
ax.set_xlabel('Tasa de intercambio K (repetidas → 1 nueva)')
ax.set_ylabel('Sobres necesarios para completar el álbum')
ax.set_title('Impacto del intercambio en el número de sobres requeridos')
ax.set_xticks(K_valores)
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('p3_intercambio.png', dpi=120, bbox_inches='tight')
plt.show()


S_valores = [5, 6, 7, 8, 9, 10]
resultados_s = {}

for sv in S_valores:
    arr = simular_lote(n_sim=2000, s=sv)
    resultados_s[sv] = arr
    media_sob = arr.mean()
    media_q   = media_sob * P_SOBRE
    print(f"S={sv} → sobres promedio: {media_sob:7.1f}  | costo estimado: Q{media_q:9.2f}")


fig, axes = plt.subplots(1, 2, figsize=(13, 5))

medias_s = [resultados_s[sv].mean() for sv in S_valores]
costos_s = [m * P_SOBRE for m in medias_s]

axes[0].bar([str(sv) for sv in S_valores], medias_s, color='mediumseagreen', edgecolor='white')
axes[0].set_xlabel('Estampas por sobre (S)'); axes[0].set_ylabel('Sobres promedio')
axes[0].set_title('Sobres necesarios según S')
for i, v in enumerate(medias_s):
    axes[0].text(i, v + 30, f'{v:.0f}', ha='center', fontsize=9)

axes[1].bar([str(sv) for sv in S_valores], costos_s, color='coral', edgecolor='white')
axes[1].set_xlabel('Estampas por sobre (S)'); axes[1].set_ylabel('Costo estimado (Q)')
axes[1].set_title('Costo estimado para completar el álbum según S')
for i, v in enumerate(costos_s):
    axes[1].text(i, v + 200, f'Q{v:.0f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('p4_estampas_sobre.png', dpi=120, bbox_inches='tight')
plt.show()


umbrales = [0.80, 0.85, 0.90, 0.95, 0.99, 1.00]
N_SIM_U = 2000

tabla = []
for u in umbrales:
    arr = simular_lote(n_sim=N_SIM_U, objetivo=u)
    tabla.append({
        'Umbral': f'{u*100:.0f}%',
        'Estampas distintas': int(u * N),
        'Sobres (media)': arr.mean(),
        'Sobres (P50)': np.median(arr),
        'Sobres (P95)': np.percentile(arr, 95),
        'Costo medio (Q)': arr.mean() * P_SOBRE,
    })
    print(f"{u*100:5.0f}% → media={arr.mean():7.1f} sobres | P95={np.percentile(arr,95):7.0f} sobres | costo medio=Q{arr.mean()*P_SOBRE:8.2f}")


import matplotlib.patches as mpatches

medias_u  = [d['Sobres (media)'] for d in tabla]
p95_u     = [d['Sobres (P95)']   for d in tabla]
etiquetas = [d['Umbral']         for d in tabla]

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(etiquetas))
w = 0.4
ax.bar(x - w/2, medias_u, width=w, label='Media', color='steelblue')
ax.bar(x + w/2, p95_u,    width=w, label='Percentil 95', color='tomato')
ax.set_xticks(x); ax.set_xticklabels(etiquetas)
ax.set_xlabel('Porcentaje del álbum completado')
ax.set_ylabel('Número de sobres')
ax.set_title('Sobres necesarios según umbral de completitud (media vs P95)')
ax.legend(); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('p5_umbrales.png', dpi=120, bbox_inches='tight')
plt.show()


