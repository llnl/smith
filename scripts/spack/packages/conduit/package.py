import os

from spack.package import *
from spack_repo.builtin.packages.conduit.package import Conduit as BuiltinConduit

class Conduit(BuiltinConduit):
    version("0.9.5", sha256="d93294efbf0936da5a27941e13486aa1a04a74a59285786a2303eed19a24265a")
    version("0.9.4", sha256="c9edfb2ff09890084313ad9c2d83bfb7c10e70b696980762d1ae1488f9f08e6c")
    version("0.9.3", sha256="2968fa8df6e6c43800c019a008ef064ee9995dc2ff448b72dc5017c188a2e6d4")
