import os

lstm_size = 1024512
port=6661
wfs = ["Montage_25", "Montage_50"]
epochs=10
episodes = 5000
for e in range(epochs):
    for wf in wfs:
        print("epoch {} wf {}".format(e, wf))
        run_name_train = "lstm_{}_e_{}_wf_{}_train".format(lstm_size,e, wf)
        run_name_test = "lstm_{}_e_{}_wf_{}_test".format(lstm_size,e, wf)
        os.system("python episode.py --is-lstm-agent=True --run-name={} --wfs-name={} --port={} --num-episodes={}".format(run_name_train, wf, port, episodes))
        os.system("python episode.py --is-lstm-agent=True --run-name={} --wfs-name={} --port={} --is-test=True".format(run_name_test, wf, port))


# fc_size = 20481024512
# port = 6661
# wfs = ["Montage_25", "CyberShake_30", "Montage_50", "CyberShake_50", "Epigenomics_24"]
# epochs=10
# episodes = 5000
# for e in range(epochs):
#     for wf in wfs:
#         print("epoch {} wf {}".format(e, wf))
#         run_name_train = "fc_{}_e_{}_wf_{}_train".format(fc_size,e, wf)
#         run_name_test = "fc_{}_e_{}_wf_{}_test".format(fc_size,e, wf)
#         os.system("python episode.py --is-lstm-agent=False --run-name={} --wfs-name={} --port={} --num-episodes={}".format(run_name_train, wf, port, episodes))
#         os.system("python episode.py --is-lstm-agent=False --run-name={} --wfs-name={} --port={} --is-test=True".format(run_name_test, wf, port))