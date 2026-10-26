/* plan.c -- 前瞻与拐角限速。纯算术，无内存分配，无 I/O。
 *
 * 算法与 Python 侧的 plan.py 是同一份数学，写在两个语言里是故意的：
 * 两个实现互相不知道对方怎么写的，唯一的共同点是公式。
 * 如果它们算出来的每一段速度都一样（bench/pair.py 去量），
 * 那就是对"实现"的一次独立检查；如果不一样，错的那个才是真正的问题。
 */
#include "cncrt.h"
#include <math.h>

static double dot3(const double a[3], const double b[3])
{
    double c = a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
    double n1 = sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
    double n2 = sqrt(b[0]*b[0] + b[1]*b[1] + b[2]*b[2]);
    if (n1 <= 0.0 || n2 <= 0.0) return 1.0;
    c /= (n1 * n2);
    if (c >  1.0) c =  1.0;
    if (c < -1.0) c = -1.0;
    return c;
}

double cncrt_corner_speed(const double t_prev[3], const double t_next[3],
                          double a_max, double jerk_dev)
{
    double c = dot3(t_prev, t_next);
    double half = 0.5 * (1.0 - c);          /* = sin^2(转弯半角) */
    if (half <= 1e-12) return -1.0;         /* -1 = 直行，不设限 */
    return sqrt(a_max * jerk_dev / half);
}

/* 一段自己允许的速度上限：圆弧有向心约束，直线没有。 */
static double seg_cap(const cncrt_seg *s, const cncrt_params *p)
{
    /* s->feed 已经由 Python 侧换算成 mm/s 传进来；这里不再做单位换算，
     * 只做上限裁剪。单位换算是解析层的责任，内核不该猜。 */
    double v = s->feed > 0.0 ? s->feed : p->v_max;
    if (v > p->v_max) v = p->v_max;
    if (s->kind == 1 && s->radius > 0.0) {
        double vc = sqrt(p->a_max * s->radius);
        if (vc < v) v = vc;
    }
    return v;
}

int cncrt_plan(const cncrt_seg *segs, size_t n, const cncrt_params *p,
               cncrt_plan_row *rows, double *node_vel, const char **binding)
{
    size_t i;
    double a, dev;
    double hi[CNCRT_MAX_SEG + 1];
    double lo[CNCRT_MAX_SEG + 1];

    if (!segs || !p || n == 0 || n > CNCRT_MAX_SEG) return -1;
    if (!rows || !node_vel || !binding) return -1;
    a = p->a_max;
    dev = p->jerk_dev;
    if (a <= 0.0 || dev <= 0.0 || p->v_max <= 0.0) return -1;

    /* 节点速度：i 是第 i 段之前那个交接处，n 是程序终点 */
    node_vel[0] = p->v_start;
    binding[0]  = "program_start";
    node_vel[n] = p->v_end;
    binding[n]  = "program_end";
    for (i = 1; i < n; i++) {
        double v = cncrt_corner_speed(segs[i-1].t_out, segs[i].t_in, a, dev);
        if (v < 0.0) {
            double c = dot3(segs[i-1].t_out, segs[i].t_in);
            rows[i].turn_deg = 0.0;
            binding[i] = "straight";
            node_vel[i] = 0.0;      /* 0 作哨兵：不设限，真正的上限由 cap 给 */
            (void)c;
        } else {
            double c = dot3(segs[i-1].t_out, segs[i].t_in);
            double ang = acos(c);
            if (ang < 0.0) ang = 0.0;
            node_vel[i] = v;
            binding[i] = "corner";
            rows[i].turn_deg = ang * 180.0 / 3.14159265358979323846;
        }
    }

    /* limit_at：第 i 个节点自己的上限 = 过弯速度、两侧段自身上限，三者取小。
     * 这一步单独写成函数，是因为原来把它写在了反推循环里，并且误用了
     * 远端节点的限速当近端节点的上限——于是"下一个弯要慢"被读成了
     * "这个弯也必须一样慢"，整条程序被过度约束，与 Python 侧差了整整 9.5 mm/s。 */
    #define LIMIT_AT(i, out) do { \
        double _c[3]; int _k = 0; \
        if (node_vel[i] > 0.0) _c[_k++] = node_vel[i]; \
        if ((i) > 0)   _c[_k++] = seg_cap(&segs[(i)-1], p); \
        if ((i) < n)   _c[_k++] = seg_cap(&segs[(i)], p); \
        if (_k > 0) out = _c[0]; else out = p->v_max; \
        { int _j; for (_j = 1; _j < _k; _j++) if (_c[_j] < out) out = _c[_j]; } \
    } while (0)

    /* 反推：在上限之内，能从下一个节点减速到的最大速度 */
    hi[n] = node_vel[n];
    for (i = n; i > 0; i--) {
        size_t j = i - 1;
        double lim, reach;
        /* 注意这里是 LIMIT_AT(j) 而不是 LIMIT_AT(i)：算出来的是第 j 个
         * 节点的入口速度，就该受第 j 个节点自己的上限约束。写成 i 就是把
         * 下游那个弯的限速强加给上游——差一位，代价是整条程序过度减速。 */
        LIMIT_AT(j, lim);
        reach = sqrt(hi[i]*hi[i] + 2.0 * a * segs[j].length);
        hi[j] = lim < reach ? lim : reach;
    }
    {
        double l0;
        LIMIT_AT(0, l0);
        if (hi[0] > l0) hi[0] = l0;
        if (hi[0] > p->v_start) hi[0] = p->v_start;
    }

    /* 正推：从上一个节点加速能到多少 */
    lo[0] = hi[0];
    for (i = 0; i < n; i++) {
        double reach = sqrt(lo[i]*lo[i] + 2.0 * a * segs[i].length);
        lo[i+1] = hi[i+1] < reach ? hi[i+1] : reach;
    }

    for (i = 0; i < n; i++) {
        double ve = lo[i] < hi[i] ? lo[i] : hi[i];
        double vx = lo[i+1] < hi[i+1] ? lo[i+1] : hi[i+1];
        double L = segs[i].length;
        double vm, d_acc, d_dec, d_cru, t;
        vm = sqrt((2.0 * a * L + ve*ve + vx*vx) / 2.0);
        {
            double v_cap_here = seg_cap(&segs[i], p);
            if (vm > v_cap_here) vm = v_cap_here;   /* 峰值不得超过本段上限 */
        }
        d_acc = (vm*vm - ve*ve) / (2.0 * a);
        d_dec = (vm*vm - vx*vx) / (2.0 * a);
        if (d_acc + d_dec > L && d_acc + d_dec > 1e-12) {
            double scale = L / (d_acc + d_dec);
            d_acc *= scale;
            d_dec *= scale;
            vm = sqrt(ve*ve + 2.0 * a * d_acc);
        }
        d_cru = L - d_acc - d_dec;
        if (d_cru < 0.0) d_cru = 0.0;
        t = 0.0;
        if (d_acc > 1e-9) t += 2.0 * d_acc / (ve + vm);
        if (d_dec > 1e-9) t += 2.0 * d_dec / (vx + vm);
        if (d_cru > 1e-9 && vm > 1e-9) t += d_cru / vm;
        if (t <= 0.0) t = 1e-12;
        rows[i].line = segs[i].line;
        rows[i].kind = segs[i].kind;
        rows[i].length = L;
        rows[i].feed = segs[i].feed;
        rows[i].v_entry = ve;
        rows[i].v_peak = vm;
        rows[i].v_exit = vx;
        rows[i].t_s = t;
        rows[i].d_acc = d_acc;
        rows[i].limit_by = binding[i] ? binding[i] : "cap";
    }
    return 0;
}

void cncrt_interp_init(cncrt_interp *it, const cncrt_plan_row *rows, size_t n)
{
    it->rows = rows;
    it->n = n;
    it->cur = 0;
    it->s = 0.0;
    it->t = 0.0;
    it->done = 0;
    it->a_seg = 0.0;
}

/* 一次 tick：按解析梯形曲线推进，输出的位置是**指令位置**。
 * 这里不模拟伺服——伺服滞后是物理，属于 plant，不属于实时环。 */
int cncrt_run_tick(cncrt_interp *it, const cncrt_seg *segs, double dt,
                   double pos_out[3], double vel_out[3])
{
    if (it->done) return 0;
    while (it->cur < it->n) {
        const cncrt_plan_row *r = &it->rows[it->cur];
        const cncrt_seg *sg = &segs[it->cur];
        double L = r->length;
        double ve = r->v_entry, vm = r->v_peak, vx = r->v_exit;
        double d_acc = r->d_acc, a, v;

        if (L <= 1e-12) { it->cur++; continue; }
        if (d_acc > L) d_acc = L;
        if (d_acc < 0.0) d_acc = 0.0;
        /* 加速度由规划器填的 d_acc 反算：vm^2 = ve^2 + 2*a*d_acc。
         * 这正是原来那个 bug 的位置——从"被裁剪后的峰值"反推加速度，
         * 在峰值被 v_max 夹住时会得到 a≈0，速度恒为 0，于是原地打转。 */
        a = (d_acc > 1e-12) ? (vm*vm - ve*ve) / (2.0 * d_acc) : 0.0;

        if (it->s < d_acc)
            v = sqrt(ve*ve + 2.0 * a * it->s);
        else if (vx > 1e-12)
            v = sqrt(vx*vx + 2.0 * a * (L - it->s));
        else
            v = vm;
        if (v < 0.0) v = 0.0;
        /* 速度退化到 0 而弧长还没走完，只可能是规划表本身退化了（例如一段
         * 长度为零、或者进出口速度都是 0 且段长为 0）。这里显式判死，
         * 不允许"零速度原地打转"——它在实时环里表现为一次永不返回的调用，
         * 在函数外的表现是整台机床停在那里而看门狗还没喂。 */
        if (v <= 1e-12) {
            it->cur++;
            it->s = 0.0;
            continue;
        }
        it->s += v * dt;
        it->t += dt;
        if (it->s >= L) {
            it->s -= L;
            it->cur++;
            if (it->cur >= it->n) { it->done = 1; return 0; }
            continue;
        }
        {
            double frac = it->s / L;
            if (sg->kind == 1 && sg->radius > 0.0) {
                double a0 = atan2(sg->start[1] - sg->centre[1], sg->start[0] - sg->centre[0]);
                double sweep = L / sg->radius;
                double ang = a0 + (sg->ccw ? sweep * frac : -sweep * frac);
                pos_out[0] = sg->centre[0] + sg->radius * cos(ang);
                pos_out[1] = sg->centre[1] + sg->radius * sin(ang);
                pos_out[2] = sg->start[2] + (sg->end[2] - sg->start[2]) * frac;
                vel_out[0] = -sin(ang) * v;
                vel_out[1] =  cos(ang) * v;
                vel_out[2] = (sg->end[2] - sg->start[2]) * v / L;
            } else {
                int k;
                double n = 0.0;
                for (k = 0; k < 3; k++) {
                    pos_out[k] = sg->start[k] + (sg->end[k] - sg->start[k]) * frac;
                    n += (sg->end[k] - sg->start[k]) * (sg->end[k] - sg->start[k]);
                }
                n = sqrt(n);
                for (k = 0; k < 3; k++)
                    vel_out[k] = n > 1e-12 ? v * (sg->end[k] - sg->start[k]) / n : 0.0;
            }
        }
        return 1;
    }
    it->done = 1;
    return 0;
}

/* 一阶位置环 x'' + kv x' = kv v_ref 的稳态跟随误差（常加速度下）：
 * 对纯速度前馈的 P 环，误差 e = v_ref / kv + a_ref / kv^2。
 * 前一项是速度滞后（可以被前馈吃掉），后一项是加速度滞后（吃不掉）。
 * 这里只返回加速度那一项——因为报告只声称那一项，不声称整台机床。 */
double cncrt_follow_err_analytic(double v, double a, double kv)
{
    (void)v;
    if (kv <= 0.0) return 0.0;
    return a / (kv * kv);
}

const char *cncrt_version(void) { return "cncrt 0.1.0 (stdlib C, no RTOS)"; }
