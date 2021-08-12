import streamlit as st
import pandas as pd
import pickle
import scipy.sparse
from sklearn.neighbors import NearestNeighbors


st.title("BOOK TALK")
books= pickle.load(open('booknew.pkl','rb'))
cosine_sim = pickle.load(open('simnew.pkl','rb'))
rating_pivot = pickle.load(open('rating_pivot2.pkl','rb'))
rating_matrix = scipy.sparse.load_npz('rat_matrix.npz')

book_name=st.selectbox('Enter a book you enjoyed reading: ', books['title'].values)
#get the index of the book
ind=books.index[books['title'] == book_name].tolist()[0]#index from the books7k

def recommend_cb(ind, cosine_sim = cosine_sim):
    recommended_books = []
    idx = ind # to get the index of the movie title matching the input movie
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)
    recommended_books = list(score_series.iloc[1:11].index)   # to get the indices of top 10 most similar books
    # [1:11] to exclude 0 (index 0 is the input movie itself)
    return recommended_books
#for the collaborative appproach
model_knn=NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=9)
model_knn.fit(rating_matrix)

def recommend_cf(idx, data, model, n_recommendations ):
    rec_ind=list()
    model.fit(data)
    isbn = books['isbn10'][idx]
    isbn_ind = rating_pivot.index[rating_pivot['ISBN'] == isbn].tolist()[0]#this index matches the books in rating_matrix
    #i.e the books being recommended so, get isbn from here, match to books
    distances, indices=model.kneighbors(data[isbn_ind], n_neighbors=n_recommendations)
    ind=indices.flatten()#these indexes don't correspond to books so can't use them directly
    for i in ind:
        if(i!=isbn_ind):
            isbn_i=books.index[rating_pivot['ISBN'][i] == books['isbn10']].tolist()[0]
            rec_ind.append(isbn_i)
    return rec_ind


#display the details of the book selected(name, image, author, description atleast

if st.button("what should I read next?"):
    st.header('SELECTED BOOK: ')
    col1, col2 = st.columns(2)
    with col1:
        st.image(books['thumbnail'][ind])
    with col2:
        st.text(books['title'][ind])
        st.text(books['authors'][ind])
        with st.expander('description'):
            st.write(books['description'][ind])
    reclist=recommend_cb(ind)
    st.header('SIMILAR BOOKS: ')
    for i in range(5):
        col1, col2 = st.columns(2)
        with col1:
            st.text(books['title'][reclist[2*i]])
            st.image(books['thumbnail'][reclist[2*i]])
            with st.expander('know more'):
                st.text('Author:')
                st.text(books['authors'][reclist[2*i]])
                st.markdown('**description**')
                st.write(books['description'][reclist[2*i]])
        with col2:
            st.text(books['title'][reclist[2*i+1]])
            st.image(books['thumbnail'][reclist[2*i+1]])
            with st.expander('know more'):
                st.text('Author:')
                st.text(books['authors'][reclist[2*i+1]])
                st.markdown('**description**')
                st.write(books['description'][reclist[2*i+1]])
    reclist_cf = recommend_cf(ind, rating_matrix, model_knn, 9)
    st.header('READERS ALSO LIKED: ')
    for i in range(4):
        col1, col2 = st.columns(2)
        with col1:
            st.text(books['title'][reclist_cf[2 * i]])
            st.image(books['thumbnail'][reclist_cf[2 * i]])
            with st.expander('know more'):
                st.text('Author:')
                st.text(books['authors'][reclist_cf[2 * i]])
                st.markdown('**description**')
                st.write(books['description'][reclist_cf[2 * i]])
        with col2:
            st.text(books['title'][reclist_cf[2 * i + 1]])
            st.image(books['thumbnail'][reclist_cf[2 * i + 1]])
            with st.expander('know more'):
                st.text('Author:')
                st.text(books['authors'][reclist_cf[2 * i + 1]])
                st.markdown('**description**')
                st.write(books['description'][reclist_cf[2 * i + 1]])

    #resdf=df[df['isbn10'].isin(recommended_books)]
