import numpy as np
import cv2


File_Name = "2_small_c_target.png"
Image_Matrix = np.array(cv2.imread(File_Name, cv2.IMREAD_GRAYSCALE))
Image_Matrix = Image_Matrix - 128
Size_Linear = Image_Matrix.shape[0] * Image_Matrix.shape[1]
Vary_Vector = np.ones((1, Size_Linear))
Mask_Vector = np.ones((1, Size_Linear))
Top_Value = 0
QUBO_Matrix = np.zeros((Size_Linear, Size_Linear))
Image_Matrix_Inv = Image_Matrix * -1
Image_Vector_Inv = np.resize(Image_Matrix_Inv, (1, Size_Linear))

for i in range(Size_Linear):
    for j in range(Size_Linear):
        if j == i:
            QUBO_Matrix[i][j] = 1
        if j != i:
            QUBO_Matrix[i][j] = Image_Vector_Inv[0][i] * Image_Vector_Inv[0][j]
print(QUBO_Matrix)
QUBO_Matrix.dump('QUBO.pkl')
for num in range(2**(Size_Linear - 1), 2**Size_Linear):
    bin_num = format(num, 'b')
    print(bin_num)
    for k in range(Size_Linear - 1):
        if bin_num[k] == '1':
            Vary_Vector[0][k] = 1
        if bin_num[k] == '0':
            Vary_Vector[0][k] = -1
    Value = np.dot(np.dot(Vary_Vector, QUBO_Matrix), Vary_Vector.transpose())
    if Value > Top_Value:
        Top_Value = Value
        Mask_Vector = Vary_Vector.copy()

Mask_Matrix = np.reshape(Mask_Vector, (Image_Matrix.shape[0], Image_Matrix.shape[1]))
print(Top_Value, Mask_Vector)
print(Mask_Matrix)




