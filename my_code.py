class pca:
    
    
    def __init__(self,X):
        import numpy as np
        self.X_meaned = X - np.mean(X,axis=0)

        self.X_scaled = self.X_meaned / np.std(self.X_meaned , axis=0)

        
    
    
    def fit(self,n_components):
        
        

     #　分散共分散行列の作成 rowver=Falseだと、各行が特徴量であると意味している今回は列に特徴量があるので、Falseとなる\n",
        import numpy as np
        cov_mat = np.cov(self.X_scaled , rowvar=False)
           #　固有値固有値ベクトルを分解\n",
        eig_value,eig_vectors =np.linalg.eigh(cov_mat)

         #固有値が大きい順にソートを行う\n",
        sorted_index = np.argsort(eig_value)[::-1]

         #固有ベクトルをソートした固有値のインデックスを用いてソートを行う
        sorted_vectors= eig_vectors[:,sorted_index]
        sorted_values = eig_value[sorted_index]
         # n_components分だけ列を取り出してくる\n",
        sorted_vectors[:,n_components]

        new_data = np.dot(self.X_scaled , sorted_vectors[:,:n_components])

        return new_data
