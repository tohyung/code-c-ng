du_lieu_x = [1, 2, 3, 4, 5]
du_lieu_y = [2, 4, 6, 8, 10]

n = len(du_lieu_x)

he_so_goc = 0.0
do_lech_b = 0.0 
learning_rate = 0.01
n = 1000


for i in range(n):
    
    tong_dao_ham_w = 0
    tong_dao_ham_b = 0
    

    for i in range(n):
        du_doan = he_so_goc * du_lieu_x[i] + do_lech_b
        sai_so = du_doan - du_lieu_y[i]
        
        tong_dao_ham_w += sai_so * du_lieu_x[i]
        tong_dao_ham_b += sai_so
    

    dao_ham_w = (2/n) * tong_dao_ham_w
    dao_ham_b = (2/n) * tong_dao_ham_b
    

    he_so_goc = he_so_goc - learning_rate * dao_ham_w
    do_lech_b = do_lech_b - learning_rate * dao_ham_b


print("Hệ số góc w =", round(he_so_goc, 4))
print("Độ lệch b =", round(do_lech_b, 4))
