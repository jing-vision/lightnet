#ifndef FACE_IO_H
#define FACE_IO_H

#ifdef __cplusplus
extern "C"{
#endif

#define MAX_NUM_USER 100
#define DISTANCE_TH 0.7f
#define BUFLEN 64               // limit of name length, nothing special
#define NUM_EMB_EACH_USER 3     // number of embeddings for each user
#define KNN_NUM 10              // KNN's K
#define MAX_NUM_RECOGNITION 3   // max number of face to recognize simultaneously

int get_num_user
    (
    void
    );

const char *get_username_by_idx
    (
    const int idx
    );

const char *get_new_username
    (
    void
    );

unsigned char add_newuser
    (
    void
    );

void save_embeddings
    (
    const char *usrname,
    const float *alldata,
    const int LEN
    );

void load_embeddings
    (
    const int LEN
    );

int run_embeddings_knn
    (
    const float *src,
    const int LEN,
    float *confidence
    );

void free_embedddings
    (
    void
    );

#ifdef __cplusplus
}
#endif

#endif // FACE_IO_H
