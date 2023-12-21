# from pyLRT import RadTran, get_lrt_folder
#
# LIBRADTRAN_FOLDER = get_lrt_folder()
#
# # slrt = RadTran(r'C:\Users\tarmstro\Python\par_qc\libRadtran-2.0.4'.strip())
# slrt = RadTran(r'C:\Users\tarmstro\Python\par_qc\libRadtran-2.0.4')
#
# slrt.options['rte_solver'] = 'disort'
# slrt.options['source'] = 'solar'
# slrt.options['wavelength'] = '200 2600'
# output = slrt.run(verbose=True)


import subprocess

test_dir = r"\\wsl.localhost\Ubuntu\home\tarmstro\libRadtran-2.0.4"

subprocess.call([r"\\wsl.localhost\Ubuntu\home\tarmstro\libRadtran-2.0.4\bin\uvspec <\\wsl.localhost\Ubuntu\home\tarmstro\libRadtran-2.0.4\examples\inp_intro.txt", '-Command', 'ls'])

# print(os.listdir(test_dir))
# (test_dir + "\bin\uvspec.sh")