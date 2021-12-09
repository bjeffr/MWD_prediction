import random
import os
import pandas as pd
from tqdm import trange


def main():
    n_samples = 5
    power = None

    # a-PS @180C
    M_K = 720.0
    M_e = 12870
    G0 = 220000.0
    tau_e = 2.20e-4
    tau_G = 1.30e-9
    G_G = 1.20e9
    beta_G = 0.390

    # Frequency outputs
    w_min = 1.0e-6
    w_max = 1.0e6
    w_int = 1.5

    comp_mode = 0

    # parameters in the model
    Z_Rouse = 1.5
    Z_Disent = 1.0

    alpha = 1.0
    t_CR_Start = 1.0
    DeltaCRInf = 0.30
    B_zeta = 2.0
    A_eq = 2.0
    B_eq = 10.0
    C_a = 0.189
    K_R = 1.664
    t_0 = 1.0e-3
    t_int = 1.02

    data = []
    for _ in trange(n_samples):
        valid = False
        while not valid:
            m_w_1 = random.randint(3 * M_e, 1000 * M_e)
            pdi_1 = random.uniform(1.01, 10.0)
            m_w_2 = random.randint(3 * M_e, 1000 * M_e)
            pdi_2 = random.uniform(1.01, 10.0)

            pdi_max = pdi_1 if pdi_1 > pdi_2 else pdi_2
            if m_w_1 > m_w_2:
                if power is None or (m_w_1 / m_w_2) > pdi_max ** power:
                    m_w_s, pdi_s = m_w_2, pdi_2
                    m_w_l, pdi_l = m_w_1, pdi_1
                    valid = True
            else:
                if power is None or (m_w_2 / m_w_1) > pdi_max ** power:
                    m_w_s, pdi_s = m_w_1, pdi_1
                    m_w_l, pdi_l = m_w_2, pdi_2
                    valid = True
        phi_l = random.uniform(0.0, 1.0)

        with open(file='inp.dat', mode='w') as f:
            f.writelines([
                f'{comp_mode}\n',
                f'{w_min} {w_max} {w_int}\n',
                f'{Z_Rouse} {Z_Disent}\n',
                f'{alpha} {t_CR_Start} {DeltaCRInf} {B_zeta} {A_eq} {B_eq}\n',
                f'{C_a} {K_R}\n',
                f'{t_0} {t_int}\n',
                f'{M_K} {M_e} {G0} {tau_e}\n',
                f'{G_G} {tau_G} {beta_G}\n',
                '2\n',
                f'0 50 {1 - phi_l}\n',
                f'{m_w_s} {pdi_s}\n',
                f'0 50 {phi_l}\n',
                f'{m_w_l} {pdi_l}\n'
            ])

        os.system('LP2R.exe')

        with open('gtp.dat') as f:
            instance = [m_w_s, pdi_s, m_w_l, pdi_l, phi_l]
            loss_mod = []

            for line in f.readlines():
                values = line.split()
                instance.append(float(values[1]))
                loss_mod.append(float(values[2]))

            instance.extend(loss_mod)
        data.append(instance)

    columns = ['M_W_S', 'PDI_S', 'M_W_L', 'PDI_L', 'phi_L']
    columns.extend([f'G\'{i + 1}' for i in range(70)])
    columns.extend([f'G\'\'{i + 1}' for i in range(70)])
    df = pd.DataFrame(data, columns=columns)

    df.to_csv(f'flow_data_2c_{power}.csv', index=False)


if __name__ == '__main__':
    main()
