/* cncrt.h -- 实时侧的全部契约，就是这个头文件。
 *
 * 这个文件里没有任何动态内存、没有任何锁、没有任何 printf。
 * 理由是它可以被放进一个周期调用的循环（1 kHz 甚至更高）而不至于
 * 在某个时刻停下来等一件别的事。所有缓冲都由调用者给，长度在创建时定死。
 *
 * 三个真实的时间尺度，必须分开说，混着说就会把 30 毫秒的东西吹成 30 微秒：
 *
 *   前瞻 + 拐角限速   cncrt_plan_*      一次算几十段，几十到几百微秒，离线做
 *   插补输出          cncrt_run_tick    每周期一次，几微秒，实时做
 *   伺服跟随          cncrt_plant_step  由物理决定，不进实时环
 *
 * 这个仓库只对第一、第二层声称硬实时，第三层是模型。
 */
#ifndef CNCRT_H
#define CNCRT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CNCRT_MAX_SEG 4096

typedef struct {
    double a_max;        /* mm/s^2 */
    double jerk_dev;     /* mm, junction deviation */
    double v_max;        /* mm/s */
    double v_start;      /* mm/s, 程序起点 */
    double v_end;        /* mm/s */
} cncrt_params;

typedef struct {
    int      line;       /* 源行号，拒绝和报告都要指名它 */
    int      kind;       /* 0 = 直线, 1 = 圆弧 */
    double   start[3];
    double   end[3];
    double   feed;       /* mm/s */
    double   length;     /* mm */
    double   t_in[3];    /* 单位切向 */
    double   t_out[3];
    double   radius;     /* mm, 直线为 0 */
    int      ccw;
    double   centre[2];
} cncrt_seg;

typedef struct {
    int      line;
    int      kind;
    double   length;
    double   feed;
    double   v_entry;
    double   v_peak;
    double   v_exit;
    double   t_s;
    double   d_acc;         /* 段内加速距离，插补器用它分加速/减速两段 */
    const char *limit_by;   /* "corner" / "straight" / "cap" / "program_start" */
    double   turn_deg;
} cncrt_plan_row;

/* 规划器：一次性拿到整条程序（离线），算位并行两遍，O(n)。
 * 返回 0 表示成功；-1 表示参数非法。 */
int cncrt_plan(const cncrt_seg *segs, size_t n, const cncrt_params *p,
               cncrt_plan_row *rows, double *node_vel, const char **binding);

/* 插补器：拿计划表跑周期。tick 一次输出一个采样点。
 * 返回 1 = 还在跑，0 = 这条程序跑完了。 */
typedef struct {
    const cncrt_plan_row *rows;
    size_t   n;
    size_t   cur;
    double   s;          /* 当前段内已走的弧长 */
    double   a_seg;      /* 规划器给的段内加速度 */
    double   t;          /* 已跑的总时间 */
    int      done;
} cncrt_interp;

void cncrt_interp_init(cncrt_interp *it, const cncrt_plan_row *rows, size_t n);
int  cncrt_run_tick(cncrt_interp *it, const cncrt_seg *segs, double dt,
                    double pos_out[3], double vel_out[3]);

/* 拐角限速这一条规则单独暴露出来，好让测试能直接考它。 */
double cncrt_corner_speed(const double t_prev[3], const double t_next[3],
                          double a_max, double jerk_dev);

/* 物理见证：一阶伺服跟随的解析解，给 Python 侧的积分做对拍。
 * max_x / max_y / max_z 是指令轴上的速度包络（mm/s），kv 是位置环增益。 */
double cncrt_follow_err_analytic(double v, double a, double kv);

const char *cncrt_version(void);

#ifdef __cplusplus
}
#endif
#endif
