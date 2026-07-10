///////////////////////////////////////////////////////////////////////////////
// PhaseRegionFlags parsing/resolution tests (OC review §3.8: strto completeness,
// sentinel rejection, get_PhaseRegion uninitialized-return fix)
///////////////////////////////////////////////////////////////////////////////

#include "tycho/detail/optimal_control/core/interface_types.h"
#include "tycho/detail/optimal_control/core/optimal_control_flags.h"
#include <gtest/gtest.h>

using namespace tycho;
using namespace tycho::oc;

// strto_PhaseRegionFlag must recognize every user-facing PhaseRegionFlags
// enumerator by name, including the ones historically missing (NodalPath,
// DefectPath, DefectPairWisePath, Params).
TEST(BuilderHardening, StrToPhaseRegionCoversAllUserRegions) {
    EXPECT_EQ(strto_PhaseRegionFlag("Front"), PhaseRegionFlags::Front);
    EXPECT_EQ(strto_PhaseRegionFlag("First"), PhaseRegionFlags::Front);
    EXPECT_EQ(strto_PhaseRegionFlag("Back"), PhaseRegionFlags::Back);
    EXPECT_EQ(strto_PhaseRegionFlag("Last"), PhaseRegionFlags::Back);
    EXPECT_EQ(strto_PhaseRegionFlag("FrontandBack"), PhaseRegionFlags::FrontandBack);
    EXPECT_EQ(strto_PhaseRegionFlag("BackandFront"), PhaseRegionFlags::BackandFront);
    EXPECT_EQ(strto_PhaseRegionFlag("Path"), PhaseRegionFlags::Path);
    EXPECT_EQ(strto_PhaseRegionFlag("InnerPath"), PhaseRegionFlags::InnerPath);
    EXPECT_EQ(strto_PhaseRegionFlag("NodalPath"), PhaseRegionFlags::NodalPath);
    EXPECT_EQ(strto_PhaseRegionFlag("DefectPath"), PhaseRegionFlags::DefectPath);
    EXPECT_EQ(strto_PhaseRegionFlag("PairWisePath"), PhaseRegionFlags::PairWisePath);
    EXPECT_EQ(strto_PhaseRegionFlag("DefectPairWisePath"), PhaseRegionFlags::DefectPairWisePath);
    EXPECT_EQ(strto_PhaseRegionFlag("Params"), PhaseRegionFlags::Params);
    EXPECT_EQ(strto_PhaseRegionFlag("ODEParams"), PhaseRegionFlags::ODEParams);
    EXPECT_EQ(strto_PhaseRegionFlag("StaticParams"), PhaseRegionFlags::StaticParams);
}

// Internal sentinels are not user-selectable region names and must be
// rejected explicitly rather than falling through to the "unrecognized" path
// (both throw, but the message should be specific to the sentinel case).
TEST(BuilderHardening, StrToPhaseRegionRejectsInternalSentinels) {
    EXPECT_THROW(strto_PhaseRegionFlag("NotSet"), std::invalid_argument);
    EXPECT_THROW(strto_PhaseRegionFlag("FrontNodalBackPath"), std::invalid_argument);
    EXPECT_THROW(strto_PhaseRegionFlag("Accumulate"), std::invalid_argument);
    EXPECT_THROW(strto_PhaseRegionFlag("BlockDefectPath"), std::invalid_argument);
}

TEST(BuilderHardening, StrToPhaseRegionRejectsUnrecognizedName) {
    EXPECT_THROW(strto_PhaseRegionFlag("NotAThing"), std::invalid_argument);
}

// get_PhaseRegion must correctly resolve both RegionType alternatives (a
// direct enumerator and a string name) now that `reg` is initialized instead
// of being read uninitialized when neither alternative branch is taken.
TEST(BuilderHardening, GetPhaseRegionResolvesEnumAlternative) {
    RegionType reg_t = PhaseRegionFlags::Back;
    EXPECT_EQ(get_PhaseRegion(reg_t), PhaseRegionFlags::Back);
}

TEST(BuilderHardening, GetPhaseRegionResolvesStringAlternative) {
    RegionType reg_t = std::string("NodalPath");
    EXPECT_EQ(get_PhaseRegion(reg_t), PhaseRegionFlags::NodalPath);
}
