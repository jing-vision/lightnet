#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include "face_io.h"


static int num_user;
static char **user_names;
static float **user_embeddings;

int get_num_user
    (
    void
    )
{
    return num_user;
}

const char *get_username_by_idx
    (
    const int idx
    )
{
    if (idx == MAX_NUM_USER) //ugly
        {
        return "unknown";
        }
    else if (idx == MAX_NUM_USER + 1)
        {
        return ".";
        }
    else
        {
        return user_names[idx];
        }
}

const char *get_new_username
    (
    void
    )
{
    return user_names[num_user - 1];
}

void save_embeddings
    (
    const char *usrname,
    const float *alldata,
    const int LEN
    )
{
    FILE *fp;
    char namebuf[BUFLEN] = {'\0'};
    sprintf(namebuf, "model/%s", usrname);
    fp = fopen(namebuf, "wb");
    fwrite(alldata, sizeof(float), NUM_EMB_EACH_USER * LEN, fp);
    fclose(fp);
    user_embeddings[num_user - 1] = (float *)malloc(NUM_EMB_EACH_USER * LEN * sizeof(float));
    memcpy(user_embeddings[num_user - 1], alldata, NUM_EMB_EACH_USER * LEN * sizeof(float));
}

unsigned char add_newuser
    (
    void
    )
{
    FILE *fp;
    char namebuf[BUFLEN] = {'\0'};
    char dirbuf[BUFLEN] = {'\0'};
    int namelen = 0;
    int i = 0;
    fp = popen("zenity --entry --text \"Please enter user name\"", "r");
    fgets(namebuf, BUFLEN, fp);
    pclose(fp);
    namelen = strlen(namebuf);
    namebuf[namelen - 1] = '\0';
    for (i = 0; i < num_user; ++i)
        {
        if (strcmp(namebuf, user_names[i]) == 0)
            {
            printf("%s already registered\n", namebuf);
            return 0;
            }
        }
    fp = fopen("data/name", "a");
    fputs(namebuf, fp);
    fputs("\n", fp);
    fclose(fp);
    user_names[num_user] = (char *)malloc(namelen * sizeof(char));
    memcpy(user_names[num_user], namebuf, namelen * sizeof(char));
    ++num_user;
    sprintf(dirbuf, "data/%s", namebuf);
    mkdir(dirbuf, 0755);
    return 1;
}

void load_embeddings
    (
    const int LEN
    )
{
    FILE *fp = fopen("data/name", "r");
    char readbuf[BUFLEN] = {'\0'};
    int i = 0;
    num_user = 0;
    user_names = (char **)malloc(MAX_NUM_USER * sizeof(char *));
    user_embeddings = (float **)malloc(MAX_NUM_USER * sizeof(float *));
    while(fgets(readbuf, BUFLEN, fp) != NULL)
        {
        int usrnamelen = strlen(readbuf);
        if (usrnamelen == 1)
            {
            continue; // ignore space line
            }
        user_names[num_user] = (char *)malloc(usrnamelen * sizeof(char));
        memcpy(user_names[num_user], readbuf, usrnamelen * sizeof(char));
        user_names[num_user][usrnamelen - 1] = '\0';
        ++num_user;
        }
    fclose(fp);
    for(i = 0; i < num_user; ++i)
        {
        sprintf(readbuf, "model/%s", user_names[i]);
        fp = fopen(readbuf, "rb");
        user_embeddings[i] = (float *)malloc(NUM_EMB_EACH_USER * LEN * sizeof(float));
        int freadnum = fread(user_embeddings[i], sizeof(float), NUM_EMB_EACH_USER * LEN, fp);
        if (freadnum != NUM_EMB_EACH_USER * LEN)
            {
            printf("registered embedding and current model are inconsistent\n");
            }
        fclose(fp);
        }
}

int distance_compare
    (
    const void * a,
    const void * b
    )
{
    return ((*(float*)a - *(float*)b) > 0.0f);
}

int run_embeddings_knn
    (
    const float *src,
    const int LEN,
    float *confidence
    )
{
    int num_comparison = num_user * NUM_EMB_EACH_USER;
    // saving 1. distance and 2. related label
    float *all_distance_and_id = (float *)calloc(num_comparison << 1, sizeof(float));
    int dis_id = 0;
    int i = 0;
    int j = 0;
    int k = 0;
    int num_candidates = 0;
    float *candidates = (float *)calloc(num_comparison << 1, sizeof(float));
    int hitsum[MAX_NUM_USER] = { 0 };
    int recog_name_idx = MAX_NUM_USER + 1; // blank

    for (i = 0; i < num_user; ++i)
        {
        float *usercase = user_embeddings[i];
        for (j = 0; j < NUM_EMB_EACH_USER; ++j)
            {
            float distance = 0.0f;
            // Euclidean distance
            for (k = 0; k < LEN; ++k)
                {
                float diff = src[k] - usercase[k];
                distance += diff * diff;
                }
            distance = sqrt(distance);

            all_distance_and_id[dis_id << 1] = distance;
            all_distance_and_id[(dis_id << 1) + 1] = i;
            ++dis_id;
            usercase += LEN;
            }
        }

    qsort(all_distance_and_id, num_comparison, 2 * sizeof(float), distance_compare);
    if (num_comparison > KNN_NUM)
        {
        num_comparison = KNN_NUM;
        }

    //printf("comparison:%d\n", num_comparison);

    for (i = 0; i < num_comparison; ++i)
        {
        int idx = i << 1;
        //printf("%f\n", all_distance_and_id[idx]);
        if (all_distance_and_id[idx] < DISTANCE_TH)
            {
            candidates[idx] = all_distance_and_id[idx];
            candidates[idx + 1] = all_distance_and_id[idx + 1];
            ++num_candidates;
            }
        }

    //for (i = 0; i < num_candidates; ++i)
        //{
        //printf("%d:%f ", (int)candidates[2*i+1], candidates[2*i]);
        //}
    //printf("\n");

    if (num_candidates == 0)
        {
        recog_name_idx = MAX_NUM_USER; // unknown
        *confidence = 2.0f;
        }
    else
        {
        memset(hitsum, 0, MAX_NUM_USER * sizeof(int));
        for (i = 0; i < num_candidates; ++i)
            {
            ++hitsum[(int)candidates[(i << 1) + 1]];
            }

        //for (i = 0; i < MAX_NUM_USER; ++i)
            //{
            //printf("%d ", hitsum[i]);
            //}
        //printf("\n");

        int maxhitsum = 0;
        int maxhitidx = 0;
        float mindist = 2.0f;
        for (i = 0; i < MAX_NUM_USER; ++i)
            {
            if (maxhitsum < hitsum[i])
                {
                maxhitsum = hitsum[i];
                maxhitidx = i;
                }
            }
        for (i = 0; i < num_candidates; ++i)
            {
            int idx = i << 1;
            if ((int)candidates[idx + 1] == maxhitidx)
                {
                if (mindist > candidates[idx])
                    {
                    mindist = candidates[idx];
                    }
                }
            }
        recog_name_idx = maxhitidx; // normal user
        *confidence = mindist;
        }

    free(all_distance_and_id);
    free(candidates);
    return recog_name_idx;
}

void free_embedddings
    (
    void
    )
{
    int i = 0;
    for (i = 0; i < num_user; ++i)
        {
        free(user_names[i]);
        free(user_embeddings[i]);
        }
    free(user_names);
    free(user_embeddings);
}
