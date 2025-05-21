#include "catch.hpp"

#include <cmath>
#include <array>
#include <UT_SolidAngle.h>

TEST_CASE("Single triangle", "[solid angle][zhou]") {
    using Float = float;
    using Index = int;
    using Vector = HDK_Sample::UT_Vector3T<Float>;

    Vector vertices[3];
    vertices[0][0] = 0.0;
    vertices[0][1] = 0.0;
    vertices[0][2] = 0.0;
    vertices[1][0] = 1.0;
    vertices[1][1] = 0.0;
    vertices[1][2] = 0.0;
    vertices[2][0] = 0.0;
    vertices[2][1] = 1.0;
    vertices[2][2] = 0.0;

    std::array<Index, 3> faces{
        0, 1, 2
    };

    HDK_Sample::UT_SolidAngle<Float, Float> engine;
    engine.init(1, faces.data(), 3, vertices);

    SECTION("Coplanar but outside") {
        Vector q;
        q[0] = 0.0;
        q[1] = 0.0;
        q[2] = 0.0;
        REQUIRE(Approx(0.0) == engine.computeSolidAngle(q));
    }

    SECTION("Coplanar but very far outside") {
        Vector q;
        q[0] = 1e12;
        q[1] = 1e-12;
        q[2] = 0.0;
        REQUIRE(Approx(0.0) == engine.computeSolidAngle(q));
    }

    SECTION("Coplanar but inside") {
        Vector q;
        q[0] = 0.1;
        q[1] = 0.1;
        q[2] = 0.0;
        // TODO!
        // Warning: This case is ill-defined.  One would expect the answer to be
        // either 2*M_PI or -2*M_PI, but the code returns the average of the
        // two: 0.  This may be problematic!
        REQUIRE(Approx(0.0) == engine.computeSolidAngle(q));
    }

    SECTION("Near coplanar") {
        Vector q;
        q[0] = 0.1;
        q[1] = 0.1;
        q[2] = -1e-12;
        REQUIRE(Approx(M_PI * 2) == engine.computeSolidAngle(q));
    }

    SECTION("Near coplanar on the other side") {
        Vector q;
        q[0] = 0.1;
        q[1] = 0.1;
        q[2] = 1e-12;
        REQUIRE(Approx(-M_PI * 2) == engine.computeSolidAngle(q));
    }

    SECTION("Positive solid angle") {
        Vector q;
        q[0] = 0.0;
        q[1] = 0.0;
        q[2] = -1.0;
        const Float theta = static_cast<Float>(
                2.0 * atan(sqrt(2)) - M_PI * 0.5);
        REQUIRE(Approx(theta) == engine.computeSolidAngle(q));
    }

    SECTION("Negative solid angle") {
        Vector q;
        q[0] = 0.0;
        q[1] = 0.0;
        q[2] = 1.0;
        const Float theta = static_cast<Float>(
                2.0 * atan(sqrt(2)) - M_PI * 0.5);
        REQUIRE(Approx(-theta) == engine.computeSolidAngle(q));
    }

}
