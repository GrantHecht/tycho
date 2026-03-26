///////////////////////////////////////////////////////////////////////////////
// ODEBuilder unit tests
///////////////////////////////////////////////////////////////////////////////

#include "oc_test_utils.h"
#include <gtest/gtest.h>
#include <tycho/detail/ode_builder.h>
#include <tycho/detail/runtime_ode.h>
#include <tycho/tycho.h>

using namespace Tycho;
using namespace TychoTest;

class ODEBuilderTest : public OptimalControlTest {};

TEST_F(ODEBuilderTest, LambdaDefinition) {
    auto ode = ODEBuilder(3, 1)
                   .define([](auto &args) {
                       auto v = args.XVar(2);
                       auto theta = args.UVar(0);
                       return stack(sin(theta) * v, cos(theta) * v * (-1.0), 9.81 * cos(theta));
                   })
                   .var_names({{"x", 0}, {"y", 1}, {"v", 2}, {"theta", 4}})
                   .build();

    EXPECT_EQ(ode.xvars(), 3);
    EXPECT_EQ(ode.uvars(), 1);

    auto input = ode.make_input({{"x", 0}, {"y", 10}, {"v", 0}, {"theta", 1.0}});
    EXPECT_EQ(input.size(), 5);
    EXPECT_DOUBLE_EQ(input[1], 10.0);
    EXPECT_DOUBLE_EQ(input[4], 1.0);
}

TEST_F(ODEBuilderTest, FromExpression) {
    auto args = ODEArguments<-1, -1, -1>(3, 1, 0);
    auto v = args.segment(0, 3).coeff(2);
    auto theta = args.segment(4, 1).coeff(0);

    auto expr = stack(sin(theta) * v, cos(theta) * v * (-1.0), 9.81 * cos(theta));

    auto ode = ODEBuilder(3, 1).from(expr).build();

    EXPECT_EQ(ode.xvars(), 3);
    EXPECT_EQ(ode.uvars(), 1);
    EXPECT_FALSE(ode.has_registry());
}

TEST_F(ODEBuilderTest, BuildWithoutDefineThrows) {
    EXPECT_THROW(ODEBuilder(3, 1).build(), std::invalid_argument);
}

TEST_F(ODEBuilderTest, DoubleDefineThrows) {
    ODEBuilder builder(3, 1);
    builder.define([](auto &args) {
        auto v = args.XVar(2);
        auto theta = args.UVar(0);
        return stack(sin(theta) * v, cos(theta) * v * (-1.0), 9.81 * cos(theta));
    });

    EXPECT_THROW(builder.define([](auto &args) { return args.XVar(0); }), std::invalid_argument);
}

TEST_F(ODEBuilderTest, VarGroups) {
    auto ode = ODEBuilder(3, 1)
                   .define([](auto &args) {
                       auto v = args.XVar(2);
                       auto theta = args.UVar(0);
                       return stack(sin(theta) * v, cos(theta) * v * (-1.0), 9.81 * cos(theta));
                   })
                   .var_names({{"x", 0}, {"y", 1}, {"v", 2}, {"theta", 4}})
                   .var_group("pos", 0, 2)
                   .build();

    auto idx = ode.registry().resolve("pos");
    EXPECT_EQ(idx.size(), 2);
    EXPECT_EQ(idx[0], 0);
    EXPECT_EQ(idx[1], 1);
}

TEST_F(ODEBuilderTest, InvalidSizesThrow) {
    EXPECT_THROW(ODEBuilder(0), std::invalid_argument);
    EXPECT_THROW(ODEBuilder(-1), std::invalid_argument);
    EXPECT_THROW(ODEBuilder(3, -1), std::invalid_argument);
    EXPECT_THROW(ODEBuilder(3, 0, -1), std::invalid_argument);
}

TEST_F(ODEBuilderTest, XVarBoundsCheck) {
    EXPECT_THROW(
        ODEBuilder(3, 1).define([](auto &args) {
            return stack(args.XVar(3), args.XVar(0), args.XVar(0));
        }),
        std::invalid_argument);
}

TEST_F(ODEBuilderTest, PVarWithoutPvarsThrows) {
    EXPECT_THROW(
        ODEBuilder(2, 1).define([](auto &args) {
            return stack(args.XVar(0), args.PVar(0));
        }),
        std::invalid_argument);
}

TEST_F(ODEBuilderTest, DoubleBuildThrows) {
    ODEBuilder builder(3, 1);
    builder.define([](auto &args) {
        auto v = args.XVar(2);
        auto theta = args.UVar(0);
        return stack(sin(theta) * v, cos(theta) * v * (-1.0), 9.81 * cos(theta));
    });

    builder.build();
    EXPECT_THROW(builder.build(), std::invalid_argument);
}

TEST_F(ODEBuilderTest, VarNamesAfterBuildThrows) {
    ODEBuilder builder(3, 1);
    builder.define([](auto &args) {
        auto v = args.XVar(2);
        auto theta = args.UVar(0);
        return stack(sin(theta) * v, cos(theta) * v * (-1.0), 9.81 * cos(theta));
    });

    builder.build();
    EXPECT_THROW(builder.var_names({{"x", 0}}), std::invalid_argument);
}

TEST_F(ODEBuilderTest, VarGroupAfterBuildThrows) {
    ODEBuilder builder(3, 1);
    builder.define([](auto &args) {
        auto v = args.XVar(2);
        auto theta = args.UVar(0);
        return stack(sin(theta) * v, cos(theta) * v * (-1.0), 9.81 * cos(theta));
    });

    builder.build();
    EXPECT_THROW(builder.var_group("pos", 0, 2), std::invalid_argument);
}

TEST_F(ODEBuilderTest, DefineWrongOutputSizeThrows) {
    // Lambda returns 2 outputs but xvars=3
    EXPECT_THROW(
        ODEBuilder(3, 1).define([](auto &args) {
            return stack(args.XVar(0), args.XVar(1));
        }),
        std::invalid_argument);
}

TEST_F(ODEBuilderTest, FromWrongOutputSizeThrows) {
    auto args = ODEArguments<-1, -1, -1>(2, 1, 0);
    auto expr = stack(args.segment(0, 2).coeff(0), args.segment(0, 2).coeff(1));

    // Expression has 2 outputs / 4 inputs, but builder declares xvars=3, uvars=1 (5 inputs, 3 outputs)
    EXPECT_THROW(ODEBuilder(3, 1).from(expr), std::invalid_argument);
}

TEST_F(ODEBuilderTest, TVarAccessor) {
    auto ode = ODEBuilder(1, 0)
                   .define([](auto &args) {
                       // dx/dt = t (time-dependent ODE)
                       return args.TVar();
                   })
                   .build();

    EXPECT_EQ(ode.xvars(), 1);
    EXPECT_EQ(ode.uvars(), 0);
    EXPECT_EQ(ode.xtup_size(), 2); // x + t
}

TEST_F(ODEBuilderTest, VectorAccessors) {
    auto ode = ODEBuilder(3, 1, 1)
                   .define([](auto &args) {
                       auto x = args.XVec();
                       auto u = args.UVec();
                       auto p = args.PVec();
                       return stack(x.coeff(0) + u.coeff(0), x.coeff(1) * p.coeff(0), x.coeff(2));
                   })
                   .build();

    EXPECT_EQ(ode.xvars(), 3);
    EXPECT_EQ(ode.uvars(), 1);
    EXPECT_EQ(ode.pvars(), 1);
    EXPECT_EQ(ode.xtup_size(), 6);
}

TEST_F(ODEBuilderTest, SubSegmentAccessors) {
    // XVec(start, count), UVec(start, count), PVec(start, count)
    auto ode = ODEBuilder(5, 3, 2)
                   .define([](auto &args) {
                       // First 3 states + first 2 controls + first param
                       auto x_sub = args.XVec(0, 3);
                       auto u_sub = args.UVec(0, 2);
                       auto p_sub = args.PVec(0, 1);
                       return stack(x_sub.coeff(0) + u_sub.coeff(0), x_sub.coeff(1) + u_sub.coeff(1),
                                    x_sub.coeff(2) + p_sub.coeff(0), args.XVar(3), args.XVar(4));
                   })
                   .build();

    EXPECT_EQ(ode.xvars(), 5);
    EXPECT_EQ(ode.uvars(), 3);
    EXPECT_EQ(ode.pvars(), 2);
}

TEST_F(ODEBuilderTest, SubSegmentBoundsCheck) {
    EXPECT_THROW(
        ODEBuilder(3, 1).define([](auto &args) {
            return stack(args.XVec(0, 4).coeff(0), args.XVar(1), args.XVar(2)); // 4 > xvars=3
        }),
        std::invalid_argument);

    EXPECT_THROW(
        ODEBuilder(3, 2).define([](auto &args) {
            return stack(args.XVar(0), args.XVar(1), args.UVec(1, 2).coeff(0)); // 1+2=3 > uvars=2
        }),
        std::invalid_argument);
}

TEST_F(ODEBuilderTest, FromAfterDefineThrows) {
    ODEBuilder builder(3, 1);
    builder.define([](auto &args) {
        return stack(args.XVar(0), args.XVar(1), args.XVar(2));
    });
    auto args = ODEArguments<-1, -1, -1>(3, 1, 0);
    auto expr =
        stack(args.segment(0, 3).coeff(0), args.segment(0, 3).coeff(1), args.segment(0, 3).coeff(2));
    EXPECT_THROW(builder.from(expr), std::invalid_argument);
}

TEST_F(ODEBuilderTest, VarNamesOutOfRangeThrows) {
    ODEBuilder builder(3, 1);
    builder.define([](auto &args) {
        return stack(args.XVar(0), args.XVar(1), args.XVar(2));
    });
    EXPECT_THROW(builder.var_names({{"bad", 999}}), std::invalid_argument);
    EXPECT_THROW(builder.var_names({{"neg", -1}}), std::invalid_argument);
}

TEST_F(ODEBuilderTest, WithPVars) {
    auto ode = ODEBuilder(2, 1, 1)
                   .define([](auto &args) {
                       auto x = args.XVar(0);
                       auto y = args.XVar(1);
                       auto u = args.UVar(0);
                       auto p = args.PVar(0);
                       return stack(y, p * u - x);
                   })
                   .build();

    EXPECT_EQ(ode.xvars(), 2);
    EXPECT_EQ(ode.uvars(), 1);
    EXPECT_EQ(ode.pvars(), 1);
    EXPECT_EQ(ode.xtup_size(), 5); // 2 + 1 + 1 + 1
}
