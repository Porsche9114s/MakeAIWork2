import os, sys
import shutil
import pandas as pd
import csv

srcpath = r"projects/maiw _project2/src"
aqlpath = r"projects/maiw_project2/src/aql_tool"
aqltestpath = r"projects/maiw_project2/src/regression"

sys.path.append(srcpath)
# import visualization.visualize as vi
# from data import make_dataset as da
# from models import *
# from aql_tool import *
from aql_tool import aqltool as aqlt


def makeinpfile(modelname, lotsize, testmode=False):
    if testmode:
        pass
    else:
        input = aqlt.Aqltool(os.path.join(aqltestpath, "aqltool.inp")).readinp()
        input["harvestbatchsize"] = str(lotsize)
        with open(os.path.join(aqltestpath, "aqltool.inp"), "w") as csv_file:
            writ = csv.writer(csv_file, delimiter=";", lineterminator="\n")
            for k in input:
                writ.writerow([k, input[k]])
        # writ.close
        # df = pd.read_csv(
        #    os.path.join(aqltestpath, "aqltool.inp"),
        # )
        # df.to_csv("aqltool.inp")
        print("input file ready....")
    # print(input2)


def start_aqltool(testmode=True):
    if testmode:
        pass
    else:
        aql = aqlt.Aqltool(os.path.join(aqltestpath, "aqltool.inp"))
        print(aql.lotsize)
        # xx
        sizecode = aql.return_sizecode_gil1()
        aqlcode = aql.return_aqlcode()
    return sizecode, aqlcode


def _read_expectedfile(expectedfile):
    df = pd.read_csv(expectedfile, delimiter=",", index_col=[0, 1])
    # df =
    # with open(expectedfile,'r') as g:
    #    expect = csv.reader(expectedfile,delimiter=',',index_col)
    #    for
    return df


def check_regressiontest(sizecode, aqlcode, applesample, harvestbatch, expectedfile):
    # COMPARE REGRESSIONTEST WITH EXPECTED VALUES
    df = _read_expectedfile(expectedfile)
    res = df.loc[input["harvestbatchsize"], input["harvestbatchsize"]]
    expectsc = res["sizecode"]
    expectac = res["aqlcode"]
    print(sizecode == expectsc)
    print(aqlcode == expectac)
    return


def write_regresresult(
    filename, inputfile, sizecode_res, aqlcode_res, sizecode_tru, aqlcode_tru
):

    regrescompare = [sizecode_res, sizecode_tru, aqlcode_res, aqlcode_tru]

    input = aqlt.Aqltool(os.path.join(aqltestpath, "aqltool.inp")).readinp()
    inputparam = sorted(input.keys())
    inptable = [input[i] for i in inputparam]
    with open(filename, "a") as g:
        writ = csv.writer(g, delimiter=";", lineterminator="")
        row = [inptable] + regrescompare
        writ.writerow(row)

    # Write results regression test.


def inittest(testmode=True):
    """
    Initializing test
    - make testenvironment
    - startup batch
    """
    if testmode:
        return
    if os.path.exists(aqltestpath):
        shutil.rmtree(aqltestpath)
        # os.rmdir(aqltestpath)
    os.mkdir(aqltestpath)

    shutil.copyfile(
        os.path.join(aqlpath, "aqltool.temp"), os.path.join(aqltestpath, "aqltool.inp")
    )
    shutil.copyfile(
        os.path.join(aqlpath, "expectedresults.csv"),
        os.path.join(aqltestpath, "expectedresults.csv"),
    )


def main():

    regres_resultfile = "C:/temp/regresresult.csv"

    for lotsize in [3, 20, 400, 6000, 1000000]:
        inittest(testmode=False)
        makeinpfile(
            "test",
            lotsize,
        )
        aqlcode, sizecode = start_aqltool(testmode=False)
        check_regressiontest(aqlcode, sizecode, "raw/Test", 30, "expectedresults.csv")


if __name__ == "__main__":
    main()
