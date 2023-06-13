from __future__ import annotations
import collections
from dataclasses import dataclass, asdict
from typing import Optional, Counter, List, Iterator, Iterable
import weakref
import csv
import enum


@dataclass
class Sample:
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
   
class Purpose(enum.IntEnum):
    Classification = 0
    Testing = 1
    Training = 2

@dataclass
class KnownSample(Sample):
    species: str

class Distance:
    """거리 계산에 대한 정의"""
    def distance(self, s1: Sample, s2: Sample) -> float:
        raise NotImplementedError

class ED(Distance):
    def distance(self, s1: Sample, s2: Sample) -> float:
        return hypot(
            s1.sepal_length - s2.sepal_length,
            s1.sepal_width - s2.sepal_width,
            s1.petal_length - s2.petal_length,
            s1.petal_width - s2.petal_width,
        )

class MD(Distance):
    def distance(self, s1: Sample, s2: Sample) -> float:
        return sum(
            [
                abs(s1.sepal_length - s2.sepal_length),
                abs(s1.sepal_width - s2.sepal_width),
                abs(s1.petal_length - s2.petal_length),
                abs(s1.petal_width - s2.petal_width) ,
            ]
        )
    
class CD(Distance):
    def distance(self, s1: Sample, s2: Sample) -> float:
        return max(
            [
                abs(s1.sepal_length - s2.sepal_length),
                abs(s1.sepal_width - s2.sepal_width),
                abs(s1.petal_length - s2.petal_length),
                abs(s1.petal_width - s2.petal_width) ,
            ]
        )

class SD(Distance):
    def distance(self, s1: Sample, s2: Sample) -> float:
        return sum(
            [
                abs(s1.sepal_length - s2.sepal_length),
                abs(s1.sepal_width - s2.sepal_width),
                abs(s1.petal_length - s2.petal_length),
                abs(s1.petal_width - s2.petal_width) ,
            ]
        ) / sum(
            [
                s1.sepal_length + s2.sepal_length,
                s1.sepal_width + s2.sepal_width,
                s1.petal_length + s2.petal_length,
                s1.petal_width + s2.petal_width,
            ]
        )
class InvalidSampleError(ValueError):
    """소스 데이터 파일이 유효하지 않은 데이터 표현을 가지고 있다."""

class OutlierError(ValueError):
    """값이 예상된 범위 밖에 있다. """

@dataclass
class Hyperparameter:
    """k 및 거리 계산 알고리듬이 있는 튜닝 매개변수 집합"""

    k: int
    algorithm: Distance
    data: weakref.ReferenceType["TrainingData"]

    def classify(self, sample: Sample) -> str:
        """k-NN 알고리듬"""
        training_data = self.data()
        if not training_data:
            raise RuntimeError("No TrainingData object")
        distances: list[tuple[float, KnownSample]] = sorted(
            (self.algorithm.distance(sample, known), known)
            for known in training_data.training
        )
        k_nearest = (known.species for d, known in distances[: self.k])
        frequency: Counter[str] = collections.Counter(k_nearest)
        best_fit, *others = frequency.most_common()
        species, votes = best_fit
        return species


@dataclass
class TrainingData:
    """샘플을 로드하고 테스트하는 메서드를 가지며,
    학습 및  테스트 데이터셋을 포함한다. """
    testing: List[TestingKnownSample]
    training: List[TrainingKnownSample]
    tuning: List[Hyperparameter]
   
    
@dataclass
class TestingKnownSample(KnownSample):
    classification: Optional[str]=None

@dataclass
class TrainingKnownSample(KnownSample):
    sample: KnownSample

class BadSampleRow(ValueError):
    pass

class SampleReader:
    """
    bazdekIris.data 파일에서 속성의 순서는 iris.names를 참조하라.
    """

    target_class = Sample
    header = [
        "sepal_length", "sepal_width", 
        "petal_length", "petal_width", "class"
        ]

    def __init__(self, source: Path) -> None:
        self.source = source

    def sample_iter(self) -> Iterator[Sample]:
        target_class = self.target_class
        with self.source.open() as source_file:
            reader = csv.DictReader(source_file, self.header)
            for row in reader:
                try:
                    sample = target_class(
                        sepal_length=float(row["sepal_length"]),
                        sepal_width=float(row["sepal_width"]),
                        petal_length=float(row["petal_length"]),
                        petal_width=float(row["petal_width"]),
                    )
                except ValueError as ex:
                    raise BadSampleRow(f"Invalid {row!r}") from ex
                yield sample


class SamplePartition(List[SampleDict], abc.ABC):
    @overload
    def __init__(self, *, training_subset: float = 0.80) -> None:
        ...

    @overload
    def __init__(
            self, 
            iterable: Optional[Iterable[SampleDict]] = None,
            *,
            training_subset: float = 0.80
    ) -> None:
        ...

    def __init__(
            self, 
            iterable: Optional[Iterable[SampleDict]] = None,
            *,
            training_subset: float = 0.80
    ) -> None:
        self.training_subset = training_subset
        if iterable:
            super().__init__(iterable)
        else:
            super().__init__()

    @abc.abstractproperty
    @property
    def training(self) -> list[TrainingKnownSample]:
        ... 

    @abc.abstractproperty
    @property
    def testing(self) -> list[TestingKnownSample]:
        ...

class ShufflingSamplePartition(SamplePartition):
    def __init__(
            self,
            iterable: Optional[Iterable[SampleDict]] = None,
            *,
            training_subset: float = 0.80
        ) -> None:
            super().__init__(iterable, training_subset=training_subset)
            self.split: Optional[int] = None

    def shuffle(self) -> None:
        if not self.split:
            random.shuffle(self)
            self.split = int(len(self) * self.training_subset)

    @property
    def training(self) -> list[TrainingKnownSample]:
        self.shuffle()
        return [TrainingKnownSample(**sd) for sd in self[:self.split]]
    
    @property
    def testing(self) -> list[TestingKnownSample]:
        self.shuffle()
        return [TestingKnownSample(**sd) for sd in self[self.split]]
    
    
class DealingPartition(abc.ABC):
    @abc.abstractclassmethod
    def __init__(
        self, 
        items: Optional[Iterable[SampleDict]],
        *,
        training_subset: Tuple[int, int] = (8, 10)
    ) -> None:
        ...

    @abc.abstractmethod
    def extend(self, items: Iterable[SampleDict]) -> None:
        ...    

    @abc.abstarctmethod    
    def append(self, item: SampleDict) -> None:
        ...
    
    @property
    @abc.abstractmethod
    def training(self) -> List[TrainingKnownSample]:
        ...

    @property
    @abc.abstractmethod
    def testing(self) -> List[TestingKnownSample]:
        ...


class CountingDealingPartition(DealingPartition):
    def __init__(
            self,
            items: Optional[Iterable[SampleDict]],
            *,
            training_subset: Tuple[int, int] = (8, 10)
    ) -> None:
        self.training_subset = training_subset
        self.counter = 0
        self._training: list[TrainingKnownSample] = []
        self._testing: list[TestingKnownSample] = []
        if items:
            self.extend(items)

    def extend(self, items: Iterable[SampleDict]) -> None:
        for item in items:
            self.append(item)

    def append(self, item: SampleDict) -> None:
        n, d = self.training_subset
        if self.counter % d < n:
            self.training.append(TrainingKnownSample(**item))
        else:
            self._testing.append(TestingKnownSample(**item))
        self.counter += 1

    @property
    def training(self) -> list[TrainingKnownSample]:
        return self._training
    
    @property
    def testing(self) -> list[TestingKnownSample]:
        return self._testing
