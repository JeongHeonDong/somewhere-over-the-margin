import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import numpy as np

zero_x = np.arange(1, 48)
zero_y = np.array([
    0.33187035786630653,
    0.38149898717083053,
    0.4022619851451722,
    0.4179608372721134,
    0.4289331532748143,
    0.43585415259959487,
    0.43906144496961513,
    0.44547602970965566,
    0.450371370695476,
    0.45425388251181636,
    0.4549291019581364,
    0.4640445644834571,
    0.46573261309925723,
    0.4640445644834571,
    0.4664078325455773,
    0.46843349088453745,
    0.4665766374071573,
    0.4716407832545577,
    0.47316002700877785,
    0.47417285617825794,
    0.4738352464550979,
    0.4787305874409183,
    0.4785617825793383,
    0.47721134368669815,
    0.4777177582714382,
    0.48092505064145846,
    0.48210668467251855,
    0.48311951384199864,
    0.48548278190411887,
    0.4822754895340986,
    0.4844699527346388,
    0.4837947332883187,
    0.4890276839972991,
    0.4922349763673194,
    0.4939230249831195,
    0.4922349763673194,
    0.4952734638757596,
    0.4949358541525996,
    0.4930790006752194,
    0.4930790006752194,
    0.49577987846049965,
    0.48565158676569886,
    0.48936529372045917,
    0.49426063470627957,
    0.48818365968939903,
    0.4903781228899392,
    0.49358541525995947,
])


zero_one_x = np.arange(1, 41)
zero_one_y = np.array([
    0.3283254557731263,
    0.38031735313977044,
    0.4015867656988521,
    0.4172856178257934,
    0.4324780553679946,
    0.4326468602295746,
    0.4395678595543552,
    0.4410871033085753,
    0.4422687373396354,
    0.44935854152599597,
    0.4471640783254558,
    0.4532410533423363,
    0.4540850776502363,
    0.4525658338960162,
    0.45678595543551653,
    0.4559419311276165,
    0.46066846725185684,
    0.46353814989871706,
    0.4647197839297772,
    0.4662390276839973,
    0.4611748818365969,
    0.46303173531397707,
    0.46336934503713706,
    0.4640445644834571,
    0.4679270762997974,
    0.4660702228224173,
    0.4675894665766374,
    0.47113436866981767,
    0.4728224172856178,
    0.4696151249155976,
    0.4714719783929777,
    0.4709655638082377,
    0.47197839297771776,
    0.48092505064145846,
    0.47687373396353816,
    0.4783929777177583,
    0.47923700202565833,
    0.474679270762998,
    0.47012153950033764,
    0.4724848075624578,
])


zero_two_x = np.arange(1, 22)
zero_two_y = np.array([
    0.3330519918973666,
    0.3858879135719109,
    0.4012491559756921,
    0.41357191087103307,
    0.4285955435516543,
    0.43923024983119513,
    0.4287643484132343,
    0.43332207967589464,
    0.4338284942606347,
    0.4355165428764348,
    0.4338284942606347,
    0.4436191762322755,
    0.44547602970965566,
    0.437373396353815,
    0.44564483457123566,
    0.44058068872383527,
    0.43214044564483456,
    0.43703578663065495,
    0.4402430790006752,
    0.43315327481431465,
    0.4387238352464551,
])

plt.plot(zero_x, zero_y, label="Margin=0", linestyle="-")
plt.plot(zero_one_x, zero_one_y, label="Margin=0.1", linestyle="--")
plt.plot(zero_two_x, zero_two_y, label="Margin=0.2", linestyle=":")
plt.legend(loc="lower right")
plt.grid(visible=True)
plt.show()
