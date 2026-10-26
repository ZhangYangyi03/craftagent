/* drv.c -- 测试夹具：把 C 内核挂到命令行，读用例、写 JSON。
 * 时间只在 tick 循环里量，且量的是 tick 本身，不把初始化算进去。 */
#include "cncrt.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* 高分辨率计时：Windows 上是 QueryPerformanceCounter，POSIX 上是 clock_gettime。
 * 用两者的交集在这里，是为了这个夹具在 CI 的 Linux 和本机 Windows 上行为一致。 */
#ifdef _WIN32
#include <windows.h>
static double now_s(void)
{
    static LARGE_INTEGER f;
    LARGE_INTEGER c;
    if (f.QuadPart == 0) QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}
#else
#include <time.h>
static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}
#endif

int main(int argc, char **argv)
{
    static cncrt_seg segs[CNCRT_MAX_SEG];
    static cncrt_plan_row rows[CNCRT_MAX_SEG];
    static double node[CNCRT_MAX_SEG + 1];
    static const char *binding[CNCRT_MAX_SEG + 1];
    cncrt_params p;
    cncrt_interp it;
    double pos[3], vel[3], dt, t0, plan_s, path = 0.0, sum_t = 0.0;
    double tick_sum = 0.0, tick_max = 0.0;
    size_t n = 0, i;
    int ticks = 0, m;
    FILE *f;

    if (argc < 2) { fprintf(stderr, "usage: %s cases.txt [dt]\n", argv[0]); return 2; }
    dt = argc > 2 ? atof(argv[2]) : 0.001;
    f = fopen(argv[1], "rb");
    if (!f) { perror("open"); return 2; }

    if (fscanf(f, "%lf %lf %lf %lf %lf", &p.a_max, &p.jerk_dev, &p.v_max,
               &p.v_start, &p.v_end) != 5) { fprintf(stderr, "bad params\n"); return 2; }

    while (n < CNCRT_MAX_SEG) {
        cncrt_seg *s = &segs[n];
        int got;
        s->radius = 0.0; s->ccw = 0; s->centre[0] = 0.0; s->centre[1] = 0.0;
        got = 0;
        got += fscanf(f, "%d %d", &s->line, &s->kind);
        got += fscanf(f, " %lf %lf %lf", &s->start[0], &s->start[1], &s->start[2]);
        got += fscanf(f, " %lf %lf %lf", &s->end[0], &s->end[1], &s->end[2]);
        got += fscanf(f, " %lf %lf", &s->feed, &s->length);
        got += fscanf(f, " %lf %lf %lf", &s->t_in[0], &s->t_in[1], &s->t_in[2]);
        got += fscanf(f, " %lf %lf %lf", &s->t_out[0], &s->t_out[1], &s->t_out[2]);
        got += fscanf(f, " %lf %d %lf %lf", &s->radius, &s->ccw, &s->centre[0], &s->centre[1]);
        if (got != 20) break;   /* 2+3+3+2+3+3+4 = 20 */
        path += s->length;
        n++;
    }
    fclose(f);
    if (n == 0) { fprintf(stderr, "no segments\n"); return 2; }

    t0 = now_s();
    if (cncrt_plan(segs, n, &p, rows, node, binding) != 0) {
        fprintf(stderr, "plan refused\n");
        return 3;
    }
    plan_s = now_s() - t0;

    cncrt_interp_init(&it, rows, n);
    while (cncrt_run_tick(&it, segs, dt, pos, vel)) ticks++;

    cncrt_interp_init(&it, rows, n);
    m = ticks < 20000 ? ticks : 20000;
    for (i = 0; i < (size_t)m; i++) {
        double a = now_s(), d;
        if (!cncrt_run_tick(&it, segs, dt, pos, vel)) break;
        d = now_s() - a;
        tick_sum += d;
        if (d > tick_max) tick_max = d;
    }
    for (i = 0; i < n; i++) sum_t += rows[i].t_s;

    printf("{\"version\": \"%s\", \"segments\": %zu, \"path_mm\": %.6f, ", cncrt_version(), n, path);
    printf("\"plan_us\": %.3f, \"ticks\": %d, \"dt\": %.6f, ", plan_s * 1e6, m, dt);
    printf("\"tick_us_mean\": %.4f, \"tick_us_max\": %.4f, \"t_plan_s\": %.9f,\n",
           m > 0 ? tick_sum * 1e6 / m : 0.0, tick_max * 1e6, sum_t);
    printf(" \"rows\": [");
    for (i = 0; i < n; i++) {
        printf("%s\n  {\"line\": %d, \"kind\": %d, \"length\": %.6f, \"feed\": %.3f, "
               "\"v_entry\": %.6f, \"v_peak\": %.6f, \"v_exit\": %.6f, \"t_s\": %.9f, "
               "\"limit_by\": \"%s\", \"turn_deg\": %.4f}",
               i ? "," : "", rows[i].line, rows[i].kind, rows[i].length, rows[i].feed,
               rows[i].v_entry, rows[i].v_peak, rows[i].v_exit, rows[i].t_s,
               rows[i].limit_by ? rows[i].limit_by : "", rows[i].turn_deg);
    }
    printf("\n ]}\n");
    return 0;
}
