import cv2 as cv
import numpy as np
import os
import pandas as pd


def Preparation(img):
    height, width = np.shape(img)
    for k in range(height):
        maxi = max([img[k, j] for j in range(width)])
        for j in range(width):
            img[k, j] = maxi

    ion1, ion2, ion3, ion4 = '', '', '', ''
    for k in range(15, 25):
        ion1 += f' {img[k, 0]}'
    for k in range(54, 64):
        ion2 += f' {img[k, 0]}'
    for k in range(90, 100):
        ion3 += f' {img[k, 0]}'
    for k in range(128, 138):
        ion4 += f' {img[k, 0]}'

    dct = {'ion1': ion1, 'ion2': ion2, 'ion3': ion3, 'ion4': ion4}
    lstGlobal.append(dct)
    cv.imwrite('prepared photos/' + name, img)


obj_list = os.listdir(path="original photos")
lstGlobal = []
for l in obj_list:
    name = str(l)
    img = cv.imread('original photos/' + name, cv.IMREAD_GRAYSCALE)
    Preparation(img)

data = {'file number': [], 'file name': [], 'qubit 1 state': [], 'qubit 2 state': [], 'qubit 3 state': [], 'qubit 4 state': []}
DF = pd.DataFrame(data)

for File_Name in range(0, len(obj_list)):
    New_Line = [File_Name, obj_list[File_Name]]
    count = ''
    for Ion_Number in range(1, 5):
        Image_Matrix = np.array(list(map(float, lstGlobal[File_Name]['ion'+str(Ion_Number)][1:].split(' '))))
        Image_Matrix = Image_Matrix - 128
        Size_Linear = Image_Matrix.shape[0]
        Vary_Vector = np.ones((1, Size_Linear))
        Mask_Vector = np.ones((1, Size_Linear))
        Top_Value = 0
        QUBO_Matrix = np.zeros((Size_Linear, Size_Linear))
        Image_Matrix_Inv = Image_Matrix
        Image_Vector_Inv = np.resize(Image_Matrix_Inv, (1, Size_Linear))

        for i in range(Size_Linear):
            for j in range(Size_Linear):
                if j == i:
                    QUBO_Matrix[i][j] = 1
                if j != i:
                    QUBO_Matrix[i][j] = Image_Vector_Inv[0][i] * Image_Vector_Inv[0][j]

        QUBO_Matrix.dump('QUBO.pkl')
        for num in range(2**(Size_Linear - 1), 2**Size_Linear):
            bin_num = format(num, 'b')
            for k in range(Size_Linear):
                if bin_num[k] == '1':
                    Vary_Vector[0][k] = 1
                if bin_num[k] == '0':
                    Vary_Vector[0][k] = -1
            Value = np.dot(np.dot(Vary_Vector, QUBO_Matrix), Vary_Vector.transpose())
            if Value > Top_Value:
                Top_Value = Value
                Mask_Vector = -1 * Vary_Vector.copy() * 127 + 127

        if sum(Mask_Vector[0]) > 200:
            conclusion = "Ion Here"
            count += '1'
            New_Line.append(1)

        else:
            conclusion = "No Ion"
            count += '0'
            New_Line.append(0)

    DF.loc[len(DF.index)] = New_Line
    New_Line = [File_Name]

DF.to_csv('Data.csv')

print('Process finished')
