.. ## Copyright (c) Lawrence Livermore National Security, LLC and
.. ## other Smith Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

.. _statemanager-label:

============
StateManager
============

StateManager is a global interface for (as the name suggests) managing state - 
specifically, the following core components of a simulation:

  1. Meshes - instances of ``mfem::ParMesh``
  #. Fields - instances of ``smith::FiniteElementState``
  #. Duals - instances of ``smith::FiniteElementDual``
  #. Material state - instances of ``smith::QuadratureData<T>``


``StateManager`` acts as both a factory for creating the latter three kinds of state and a means
of restarting a simulation, where state is saved to a file and later reloaded to an identical
point in the simulation.  Critically, this abstraction means that implementers of physics modules
do not have to worry about restart vs. non-restart logic, as these factory methods can be used
identically in either case.

Under the hood, ``StateManager`` is implemented in terms of
Axom's ``axom::sidre::MFEMSidreDataCollection``, whose user documentation is available
`here <https://axom.readthedocs.io/en/develop/axom/sidre/docs/sphinx/mfem_sidre_datacollection.html>`_.

``MFEMSidreDataCollection`` is an implementation of
MFEM's `DataCollection interface <https://mfem.github.io/doxygen/html/classmfem_1_1DataCollection.html>`_,
which allows for instances of ``mfem::GridFunction`` and ``mfem::QuadratureFunction`` to be associated
with a single instance of ``mfem::Mesh``.  Because Smith supports multiple meshes within a given simulation,
``StateManager`` acts as an abstraction over multiple ``MFEMSidreDataCollection`` instances.


Nominal (non-restart) Workflow
------------------------------

The first interaction a user must always make with ``StateManager`` is a call to its static
``initialize()`` method.  ``StateManager`` is implemented as a global singleton so that its
contained data can be accessed from physics modules anywhere in the simulation.  The singleton
is initialized with a non-owning reference to an ``axom::sidre::DataStore`` and an output directory
to which data will be saved.

.. note:: ``StateManager`` does not own its ``DataStore`` because Smith uses a single datastore to
  store different kinds of data - that is, data unrelated to the state defined above.  In particular,
  input file data is also stored in the per-simulation ``DataStore`` instance.

Before any other kinds of state can be created, a mesh must be registered via ``SetMesh()``.
In order for a restart to work properly, all state data must be owned by the underlying
``StateManager``, so ownership of the mesh is transferred via a ``unique_ptr``.  In the case of multi-mesh
simulations, a name or "tag" should also be used to uniquely identify the mesh. This is done automatically via
the ``smith::Mesh`` class.

Individual physics modules - that are of course based on these kinds of state - can now be constructed.
In general, this process looks something like the following:

  1. The physics module constructor accepts a mesh pointer and forwards it to the ``BasePhysics``
     constructor. This parameter is required only in multi-mesh configurations and defaults to the
     default mesh otherwise.  Although meshes also have string-valued tags associated with them,
     a user of a physics module would find it more intuitive to pass the mesh pointer they wish to use.
     Specifically, the meaning of a mesh parameter is much easier to discern than a string parameter.

  #. The physics module creates its fields (e.g., temperature for a heat transfer module) via
     calls to ``StateManager::newState()``.  In addition to the ``smith::FiniteElementState`` constructor
     arguments, this method also accepts a string-valued tag for the mesh with which the field is
     associated.  The appropriate tag is a member of ``BasePhysics`` and initialized in the previous step.
     FIXME: Should we provide a protected helper method in ``BasePhysics`` so derived modules don't need
     to reference the member explicitly? Or perhaps ``StateManager::newState()`` et al should just take
     a mesh pointer instead of a tag?

  #. The ``FiniteElementState`` is then constructed and registered in the corresponding ``MFEMSidreDataCollection``.
     The only tricky part about this process is the need for the underlying ``GridFunction`` to be allocated within
     Sidre.  This required an additional option to the ``FiniteElementState`` constructor that leaves the vector
     data uninitialized (aka ``nullptr``).  After we create the ``FiniteElementState`` we register its underlying
     ``GridFunction`` in the ``MFEMSidreDataCollection`` and zero it out.

  #. When the user wishes to save simulation state to disk, they can call ``outputState()`` on their physics module.
     FIXME: This could be confusing because this will call the global ``StateManager::save()`` which will save all
     data associated with a particular mesh. In particular we wouldn't want users to save twice if they have two
     physics modules on the same mesh (by calling ``outputState`` on each).

The use of ``smith::QuadratureData<T>`` for material state data is discussed :ref:`here <quadraturedata-label>`.

Restart Workflow
----------------

The "metadata" ``StateManager`` uses for choosing a restart file is the cycle (aka step number).  These are used
in ``StateManager::save()`` and ``StateManager::load()`` and subsequently as part of the filename written to disk.

As with a nominal workflow, the user must call ``initialize()``.  Note that while in the nominal case the directory
parameter refers to the directory to which data will be saved, in a restart case this is also the directory from
which data will be loaded.

Because the mesh already exists in the save file from which we're restarting, there is no need to call ``setMesh()``.
Instead, the user calls ``StateManager::load()``, passing it the cycle number from which they wish to restart and
the tag identifying the mesh.  As in the nominal case, this tag is not necessary for single-mesh simulations.

.. warning:: Because the mesh tag is used in the filename, it must exactly match the tag used in the call to ``setMesh()``
  in the previous simulation run.

``StateManager::load()`` will reconstruct the ``mfem::ParMesh``, ``mfem::GridFunction``, and ``mfem::QuadratureFunction``
objects.  The ``StateManager`` factory methods can be used in the exact same way as they would in a nominal run, though
the internal logic is of course different.  In particular, it will search through the restored data for a field with the
requested name and use that instead of constructing a new field via the process described above.

.. _quadraturedata-label:

QuadratureData
--------------

Smith's ``QuadratureData`` template is an abstraction over ``mfem::QuadratureFunction``, the type used to store per-quadrature-point
data.  We implement this functionality in terms of ``mfem::QuadratureFunction`` so that we can store this data in ``MFEMSidreDataCollection``,
which implements ``mfem::DataCollection::RegisterQField`` (which accepts a ``QuadratureFunction`` ).

Because ``QuadratureFunction`` only allows for floating-point data (as either scalars or vectors), ``QuadratureData<T>`` allows
for the storage of arbitrary (user-defined) types via a double-buffer approach.  That is, data is stored in a buffer of type ``T[]``
for easy access within the ``smith::Functional`` ecosystem (which natively supports ``QuadratureData`` instances) and then copied
(via a bit_cast) to the ``double[]`` buffer encapsulated by an ``mfem::QuadratureFunction`` when we wish to save state to disk.  In the case of a
restart the process works in reverse - data is ``bit_cast`` 'ed from the ``double[]`` buffer to the ``T[]`` buffer.

To allow synchronization to occur only when necessary, the ``StateManager`` registers a reference to each ``QuadratureData`` in a
type-erased (via virtual functions) callback list.  This further layer of abstraction - called ``SyncableData`` - allows 
quadrature point data of varying types to be uniformly synchronized to the corresponding ``mfem::QuadratureFunction`` instances.
