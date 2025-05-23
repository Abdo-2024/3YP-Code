#pragma once

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/surface_point.h"
#include "geometrycentral/surface/vertex_position_geometry.h"


namespace geometrycentral {
namespace surface {

/*
  This class provides a wide variety of functionality to modify a surface mesh with low-level operations like edge
  flips, splits, face splits, edge collapses, etc., as well as high-level operations like subdivisions and remeshing.
  Callbacks can be registered

  Unlike much of geometry-central, which abstractly separates the idea of a mesh's connectivity and geometry, this class
  explicitly handles the common case of a manifold triangle mesh with 3D vertex positions.



  // TODO no reason this should be limited to manifold
*/

// Forward declare
class MutationManager;


// ======================================================
// ======== Callback classes (general case)
// ======================================================

// Parent type for all policies
class MutationPolicy {
public:
  virtual ~MutationPolicy() = default;
};

class VertexRepositionPolicy : public virtual MutationPolicy {
public:
  virtual void beforeVertexReposition(const Vertex& v, const Vector3& vec) {}
};

class EdgeFlipPolicy : public virtual MutationPolicy {
public:
  virtual void beforeEdgeFlip(const Edge& e) {}
  virtual void afterEdgeFlip(const Edge& e) {}
};

class EdgeSplitPolicy : public virtual MutationPolicy {
public:
  virtual void beforeEdgeSplit(const Edge& e, const double& tSplit) {}

  // old edge E is split to halfedge HE1,HE2 both with he.vertex() as split vertex, new vertex at double tSplit
  virtual void afterEdgeSplit(const Halfedge& he1, const Halfedge& he2, const double& tSplit) {}
};

class EdgeCollapsePolicy : public virtual MutationPolicy {
public:
  virtual void beforeEdgeCollapse(const Edge& e, const double& tCollapse) {}
  virtual void afterEdgeCollapse(const Vertex& v, const double& tCollapse) {}
};


class FaceSplitPolicy : public virtual MutationPolicy {
public:
  virtual void beforeFaceSplit(const Face& f, const std::vector<double>& bSplit) {}

  // new vertex with barycentric coordinates bSplit within face f
  virtual void afterFaceSplit(const Vertex& v, const std::vector<double>& bSplit) {}
};

class MutationPolicyHandle {
public:
  MutationPolicyHandle(MutationManager& manager, MutationPolicy* policy);
  void remove();
  MutationManager& manager;
  MutationPolicy* policy;
};


// ======================================================
// ======== The Mutation Manager
// ======================================================


class MutationManager {


public:
  // ======================================================
  // ======== Members
  // ======================================================
  //
  // Core members which define the class.

  ManifoldSurfaceMesh& mesh;
  VertexPositionGeometry* geometry = nullptr; // if the mutation-manager is connectivity-only, this field will be null

  // ======================================================
  // ======== Construtors
  // ======================================================

  // Create a new mutation manager, which modifies the mesh and geometry in-place.
  MutationManager(ManifoldSurfaceMesh& mesh, VertexPositionGeometry& geometry);

  // Create a connectivity-only mutation manager, which does not touch any geometry by default
  MutationManager(ManifoldSurfaceMesh& mesh);


  // ======================================================
  // ======== Low-level mutations
  // ====================================================== //
  // Low-level routines which modify the mesh, updating connectivity as well as geometry and invoking any relevant
  // callbacks.
  //
  // Note: every one of these operations corresponds to a callback below. We should probably be wary about adding too
  // many of these, since each adds the burden of a corresponding callback policy to update data.

  // Move a vertex in 3D space
  void repositionVertex(Vertex vert, Vector3 offset);

  // Flip an edge.
  bool flipEdge(Edge e);

  // Insert a vertex along edge e at the position specified by `tSplit`.
  // Returns the new halfedge whose tail vertex is the newly inserted vertex, and points in the same direction as
  // e.halfedge() (same as the original insertVertexAlongEdge().)
  Halfedge insertVertexAlongEdge(Edge e, double tSplit);
  // Inserts a set of vertices along edge e at the specified positions.
  // Currently just returns the set of newly inserted vertices. I can't get the halfedges version working.
  // [Ideally returns a vector of new halfedges that point along the original e.halfedge(), whose tail vertices
  // correspond to the newly-inserted vertices at the positions specified by <tSplits> (in that order.)]
  std::vector<Vertex> insertVerticesAlongEdge(Edge e, const std::vector<double>& tSplits);

  // Split an edge.
  // `tSplit` controls the split location from [0,1]; 0 splits at d.halfedge().tailVertex().
  // `newVertexPosition` can be specified to manually control the insertion location, rather than linearly interpolating
  // to `tSplit`.
  // In general, both the `tSplit` and the `newVertexPosition` are used; `tSplit` is necessary to allow callbacks to
  // interpolate data; if called with either the other will be inferred.
  // Returns the new halfedge which points away from the new vertex (so he.vertex() is the new vertex), and is the same
  // direction as e.halfedge() on the original edge. The halfedge direction of the other part of the new split edge is
  // also preserved.
  Halfedge splitEdge(Edge e, double tSplit);
  Halfedge splitEdge(Edge e, Vector3 newVertexPosition);
  Halfedge splitEdge(Edge e, double tSplit, Vector3 newVertexPosition);

  // Collapse an edge.
  // Returns the new vertex if the edge could be collapsed, and Vertex() otherwise
  Vertex collapseEdge(Edge e, double tCollapse);
  Vertex collapseEdge(Edge e, Vector3 newVertexPosition);
  Vertex collapseEdge(Edge e, double tCollapse, Vector3 newVertexPosition);

  // Split a face (i.e. insert a vertex into the face), and return the new vertex
  Vertex splitFace(Face f, const std::vector<double>& bSplit);
  Vertex splitFace(Face f, Vector3 newVertexPosition);
  Vertex splitFace(Face f, const std::vector<double>& bSplit, Vector3 newVertexPosition);

  // Cut the mesh along the two given SurfacePoints, which must share a common face. The cut must span the face (the
  // face gets split into two separate faces.) Return the new halfedge whose tail vertex is at pA.
  Halfedge cutFace(SurfacePoint pA, SurfacePoint pB);
  // Cut the mesh by connecting two vertices that share a common face. Contrary to
  // ManifoldSurfaceMesh::connectVertices(), the two vertices can be adjacent, in which case it doesn't alter the mesh
  // and just returns the halfedge that already exists from vA to vB. This function is basically a nice wrapper around
  // ManifoldSurfaceMesh::connectVertices().
  Halfedge cutFace(Vertex vA, Vertex vB);
  // Cut the mesh along the specified path, where each segment of the path lies within a face.
  // Returns the list of new halfedges which makes up the newly-inserted cut, in the order of the edges in the cut path.
  std::vector<Halfedge> cutAlongPath(const std::vector<std::vector<SurfacePoint>>& pathCurves);
  std::vector<Halfedge> cutAlongPath(const std::vector<SurfacePoint>& pathNodes,
                                     const std::vector<std::array<size_t, 2>>& pathEdges);

  // ======================================================
  // ======== High-level mutations
  // ======================================================
  //
  // High-level routines which perform many low-level operations to modify a mesh

  // Improve mesh quality by alternating between perturbing vertices in their tangent plane, and splitting/collapsing
  // poorly-sized edges.
  // void isotopicRemesh();

  // Simplify the the mesh by performing edge collapses to minimize error in the sense of the quadric error metric.
  // see [Garland & Heckbert 1997] "Surface Simplification Using Quadric Error Metrics"
  // void quadricErrorSimplify(size_t nTargetVerts);

  // Split edges in the mesh until all edges satisfy the Delaunay criterion
  // see e.g. [Liu+ 2015] "Efficient Construction and Simplification of Delaunay Meshes"
  // void splitEdgesToDelaunay();


  // ======================================================
  // ======== Callbacks
  // ======================================================
  //
  // Get called whenever mesh mutations occur. Register a callback by inserting it in to this list.

  // == Automatic interface

  // These help you keep buffers of data up to date while modifying a mesh. Internally, they register functions for all
  // of the callbacks above to implement some policy.
  //
  // TODO need some way to stop managing and remove from the callback lists. Perhaps return a token class that
  // de-registers on destruction?

  // Update scalar values at vertices,
  // MutationPolicyHandle managePointwiseScalarData(VertexData<double>& data);

  // Update a paramterization
  // MutationPolicyHandle manageParameterization(CornerData<Vector2>& uvData);

  // Update integrated data at faces (scalars which have been integrated over a face, like area)
  // MutationPolicyHandle manageIntegratedScalarData(FaceData<double>& data);

  // ======================================================
  // ======== Helpers to expose the "simple" interface
  // ======================================================

  // Use these functions to quickly register callback functions, usually defined via lambda functions, which are called
  // before (`pre`) or after (`post`) various mesh mutation operations. Optionally, a data object of any type can be
  // passed from the return value of the `pre` function to the last argument of the `post` function.
  //
  // For example, the following snippet defines an edge split callback, which passes a bool from the pre to the post
  // function. It preserves the value of a hypothetical external EdgeData array called `myArr` through edge splits.
  //
  // auto funcPre = [&](Edge oldE, double tSplit) -> bool { return myArr[oldE]; };
  // auto funcPost = [&](Halfedge newHe1, Halfedge newHe2, double tSplit, bool val) -> void {
  //   myArr[newHe1] = val;
  //   myArr[newHe2] = val;
  // };
  // mutationMananger.registerEdgeSplitHandlers(funcPre, funcPost);
  //
  // See the operations above for examples of what the arguments corresponding to each function should be. If no data is
  // being passed between the functions, both pre and post should return `void`, and there will be no additional final
  // argument for `post`.
  //
  // TODO write tests to _at least_ instantiate all of these


  template <typename Fpre, typename Fpost>
  void registerEdgeFlipHandlers(Fpre pre, Fpost post);
  template <typename Fpre>
  void registerEdgeFlipPreHandler(Fpre pre);
  template <typename Fpost>
  void registerEdgeFlipPostHandler(Fpost post);

  template <typename Fpre, typename Fpost>
  void registerEdgeSplitHandlers(Fpre pre, Fpost post);
  template <typename Fpre>
  void registerEdgeSplitPreHandler(Fpre pre);
  template <typename Fpost>
  void registerEdgeSplitPostHandler(Fpost post);

  template <typename Fpre, typename Fpost>
  void registerFaceSplitHandlers(Fpre pre, Fpost post);
  template <typename Fpre>
  void registerFaceSplitPreHandler(Fpre pre);
  template <typename Fpost>
  void registerFaceSplitPostHandler(Fpost post);

  template <typename Fpre, typename Fpost>
  void registerEdgeCollapseHandlers(Fpre pre, Fpost post);
  template <typename Fpre>
  void registerEdgeCollapsePreHandler(Fpre pre);
  template <typename Fpost>
  void registerEdgeCollapsePostHandler(Fpost post);


  // == Manual interface: add callbacks on to these lists to do whatever you want.

  // After registering a policy, geometry-central will take ownership of the memory and delete it when necessary. Call
  // `handle.remove()` to un-register the policy.
  MutationPolicyHandle registerPolicy(MutationPolicy* policyObject);

  // Remove a previously registered policy. Alternately, call handle.remove();
  void removePolicy(const MutationPolicyHandle& toRemove);

  // ======================================================
  // ======== Utilities
  // ======================================================


private:
  // ======================================================
  // ======== Callback management
  // ======================================================

  // The global list of all callback policies. Also handles memory ownership.
  std::vector<std::unique_ptr<MutationPolicy>> allPolicies;

  // The lists where callbacks are stored; these are iterated through and invoked when needed.
  std::vector<VertexRepositionPolicy*> vertexRepositionPolicies;
  std::vector<EdgeFlipPolicy*> edgeFlipPolicies;
  std::vector<EdgeSplitPolicy*> edgeSplitPolicies;
  std::vector<EdgeCollapsePolicy*> edgeCollapsePolicies;
  std::vector<FaceSplitPolicy*> faceSplitPolicies;
};

} // namespace surface
} // namespace geometrycentral

#include "geometrycentral/surface/mutation_manager.ipp"
