import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import spacy
from unstructured.partition.auto import partition


class PDFRelationExtractor:
    def __init__(self, use_ocr: bool = True):
        self.nlp = self.setup_spacy_pipeline()
        self.use_ocr = use_ocr

    def setup_spacy_pipeline(self):
        """Load spacy model with built-in parser."""
        try:
            nlp = spacy.load("en_core_web_sm")
            return nlp
        except OSError:
            print("Error: spaCy model 'en_core_web_sm' not found.")
            print("Please install it with: python -m spacy download en_core_web_sm")
            sys.exit(1)

    def extract_elements_from_pdf(self, pdf_path: str) -> List[Dict]:
        """Extract elements from PDF with enhanced OCR support."""
        try:
            # 配置OCR参数
            ocr_kwargs = {}
            if self.use_ocr:
                ocr_kwargs = {
                    "ocr_mode": "auto",  # 自动检测需要OCR的页面
                    "languages": ["eng"],  # 英语OCR
                }

            elements = partition(filename=pdf_path, strategy="auto", **ocr_kwargs)
            return self._process_elements(elements)

        except Exception as e:
            print(f"PDF processing error: {e}. Using sample data for testing.")
            return self._create_sample_elements()

    def _process_elements(self, elements) -> List[Dict]:
        """处理提取的元素，识别OCR文本"""
        element_data = []

        for i, element in enumerate(elements):
            element_type = type(element).__name__
            text_content = str(element)

            # 检测是否为OCR生成的文本
            is_ocr = False
            if hasattr(element, "metadata"):
                metadata = element.metadata
                is_ocr = getattr(metadata, "is_ocr", False)

            element_info = {
                "element_id": i,
                "text": text_content,
                "type": element_type,
                "is_ocr_text": is_ocr,
                "page_number": self._extract_page_number(element),
            }

            element_data.append(element_info)

        return element_data

    def _extract_page_number(self, element) -> int:
        """提取页码信息"""
        if hasattr(element, "metadata"):
            metadata = element.metadata
            if hasattr(metadata, "page_number") and metadata.page_number is not None:
                return metadata.page_number
            elif isinstance(metadata, dict) and "page_number" in metadata:
                return metadata["page_number"]
        return 1

    def _create_sample_elements(self) -> List[Dict]:
        """Create sample elements with page information for testing."""
        sample_texts = [
            "Systematic 13F Hedge Fund Alpha. Institutions holding greater than $100 million in securities are required to disclose their holdings to the SEC.",
            "Hedge funds produce returns that outperform the S&P500. Cohen, Polk, and Silli (2010) identified key metrics.",
            "A strategy combining conviction and consensus outperforms the S&P500 by 3.80% with Sharpe ratio of 0.75 from May 2004 to June 2019.",
        ]

        elements = []
        for i, text in enumerate(sample_texts):
            elements.append(
                {
                    "element_id": i,
                    "text": text,
                    "type": "NarrativeText",
                    "page_number": (i % 3) + 1,
                    "is_ocr_text": False,
                }
            )

        return elements

    def extract_entities_relations_with_sources(
        self, elements: List[Dict]
    ) -> Tuple[List[Dict], List[Dict]]:
        """Extract entities and relations with source tracking - ENHANCED WITH OCR SUPPORT."""
        entities = []
        relations = []

        print("Processing PDF content...")

        # 统计OCR元素
        ocr_elements = [e for e in elements if e.get("is_ocr_text", False)]
        if ocr_elements:
            print(f"📷 Detected {len(ocr_elements)} OCR-generated elements")

        for element in elements:
            text = element["text"]

            if len(text.strip()) < 10:
                continue

            # 对OCR文本进行预处理
            if element.get("is_ocr_text", False):
                text = self._preprocess_ocr_text(text)

            # Process text with spaCy
            doc = self.nlp(text)

            # Extract entities with source information
            element_entities = self._extract_entities_with_source(doc, element)
            entities.extend(element_entities)

            # Extract relations with source information
            element_relations = self._extract_relations_with_source(
                doc, element, entities
            )
            relations.extend(element_relations)

        # Remove duplicate entities
        entities = self._deduplicate_entities(entities)

        return entities, relations

    def _preprocess_ocr_text(self, text: str) -> str:
        """预处理OCR文本，提高质量"""
        # 常见的OCR错误修正
        corrections = {
            " rn ": " m ",
            " cl ": " d ",
            " I ": " l ",  # 大写I被误识别为小写l
            " O ": " 0 ",  # 字母O被误识别为数字0
            "|": "I",
            "[": "I",
            "}": "J",
            # 添加更多常见OCR错误映射
        }

        processed_text = text
        for wrong, correct in corrections.items():
            processed_text = processed_text.replace(wrong, correct)

        return processed_text

    def _extract_entities_with_source(self, doc, element: Dict) -> List[Dict]:
        """Extract entities with source information."""
        entities = []
        meaningful_types = [
            "ORG",
            "PERSON",
            "GPE",
            "MONEY",
            "PERCENT",
            "DATE",
            "LAW",
            "PRODUCT",
            "EVENT",
            "WORK_OF_ART",
            "NORP",
        ]

        for ent in doc.ents:
            if ent.label_ in meaningful_types:
                entity_data = {
                    "text": ent.text,
                    "type": ent.label_,
                    "sources": [
                        {
                            "element_id": element["element_id"],
                            "page_number": element.get("page_number", "unknown"),
                            "context": doc.text[
                                max(0, ent.start_char - 50) : min(
                                    len(doc.text), ent.end_char + 50
                                )
                            ],
                            "sentence": doc.text,
                            "start_char": ent.start_char,
                            "end_char": ent.end_char,
                            "is_ocr": element.get("is_ocr_text", False),
                        }
                    ],
                }
                entities.append(entity_data)

        return entities

    def _extract_relations_with_source(
        self, doc, element: Dict, entities: List[Dict]
    ) -> List[Dict]:
        """Extract relations with source information."""
        relations = []

        # Method 1: Verb-based relation extraction
        relations.extend(self._extract_verb_relations(doc, element, entities))

        # Method 2: Pattern-based relation extraction
        relations.extend(self._extract_pattern_relations(doc, element, entities))

        # Method 3: Co-occurrence based relations (fallback)
        if len(relations) == 0:
            relations.extend(
                self._extract_cooccurrence_relations(doc, element, entities)
            )

        return relations

    def _extract_verb_relations(
        self, doc, element: Dict, entities: List[Dict]
    ) -> List[Dict]:
        """Extract relations based on verbs in the dependency tree."""
        relations = []
        entity_texts = [e["text"] for e in entities]

        for sent in doc.sents:
            for token in sent:
                if token.pos_ == "VERB":
                    # Find subject and object in the dependency tree
                    subject = None
                    obj = None

                    # Look for subjects
                    for child in token.children:
                        if child.dep_ in ["nsubj", "nsubjpass"]:
                            for entity_text in entity_texts:
                                if (
                                    child.text in entity_text
                                    or entity_text in child.text
                                ):
                                    subject = entity_text
                                    break

                    # Look for objects
                    for child in token.children:
                        if child.dep_ in ["dobj", "pobj", "attr"]:
                            for entity_text in entity_texts:
                                if (
                                    child.text in entity_text
                                    or entity_text in child.text
                                ):
                                    obj = entity_text
                                    break

                    # If we found both subject and object, create relation
                    if subject and obj and subject != obj:
                        relation_data = {
                            "subject": subject,
                            "relation": token.lemma_,
                            "object": obj,
                            "sources": [
                                {
                                    "element_id": element["element_id"],
                                    "page_number": element.get(
                                        "page_number", "unknown"
                                    ),
                                    "context": sent.text,
                                    "sentence": sent.text,
                                    "confidence": "medium",
                                    "is_ocr": element.get("is_ocr_text", False),
                                }
                            ],
                        }
                        relations.append(relation_data)

        return relations

    def _extract_pattern_relations(
        self, doc, element: Dict, entities: List[Dict]
    ) -> List[Dict]:
        """Extract relations based on patterns."""
        relations = []

        for sent in doc.sents:
            text = sent.text.lower()

            # Financial relation patterns
            patterns = [
                (["hold", "own", "manage", "posses", "holding"], "holds"),
                (["invest", "allocate", "buy", "purchase"], "invests_in"),
                (["outperform", "beat", "exceed", "surpass"], "outperforms"),
                (["report", "disclose", "file", "submit"], "reports_to"),
                (["study", "analyze", "research", "examine"], "studied"),
                (["find", "show", "demonstrate", "prove"], "found"),
                (["create", "develop", "propose", "design"], "created"),
                (["require", "mandate", "oblige"], "requires"),
                (["produce", "generate", "yield", "deliver"], "produces"),
                (["combine", "merge", "integrate", "mix"], "combines"),
                (["identify", "detect", "recognize"], "identified"),
                (["select", "choose", "pick"], "selects"),
            ]

            for pattern_verbs, relation_type in patterns:
                if any(verb in text for verb in pattern_verbs):
                    # Find entities in this sentence
                    sent_entities = [e for e in entities if e["text"] in sent.text]

                    if len(sent_entities) >= 2:
                        # Create relations between entities in the sentence
                        for i in range(len(sent_entities)):
                            for j in range(i + 1, len(sent_entities)):
                                ent1 = sent_entities[i]
                                ent2 = sent_entities[j]

                                # Only create relation if it makes sense
                                if self._is_reasonable_relation(
                                    ent1, ent2, relation_type
                                ):
                                    relation_data = {
                                        "subject": ent1["text"],
                                        "relation": relation_type,
                                        "object": ent2["text"],
                                        "sources": [
                                            {
                                                "element_id": element["element_id"],
                                                "page_number": element.get(
                                                    "page_number", "unknown"
                                                ),
                                                "context": sent.text,
                                                "sentence": sent.text,
                                                "confidence": "pattern",
                                                "is_ocr": element.get(
                                                    "is_ocr_text", False
                                                ),
                                            }
                                        ],
                                    }
                                    relations.append(relation_data)

        return relations

    def _extract_cooccurrence_relations(
        self, doc, element: Dict, entities: List[Dict]
    ) -> List[Dict]:
        """Extract relations based on co-occurrence in sentences."""
        relations = []

        for sent in doc.sents:
            sent_entities = [e for e in entities if e["text"] in sent.text]

            if len(sent_entities) >= 2:
                # Create relations between entities in the same sentence
                for i in range(len(sent_entities)):
                    for j in range(i + 1, len(sent_entities)):
                        ent1 = sent_entities[i]
                        ent2 = sent_entities[j]

                        # Determine relation type based on entity types and context
                        relation_type = self._infer_relation_type(ent1, ent2, sent.text)

                        relation_data = {
                            "subject": ent1["text"],
                            "relation": relation_type,
                            "object": ent2["text"],
                            "sources": [
                                {
                                    "element_id": element["element_id"],
                                    "page_number": element.get(
                                        "page_number", "unknown"
                                    ),
                                    "context": sent.text,
                                    "sentence": sent.text,
                                    "confidence": "co-occurrence",
                                    "is_ocr": element.get("is_ocr_text", False),
                                }
                            ],
                        }
                        relations.append(relation_data)

        return relations

    def _is_reasonable_relation(
        self, ent1: Dict, ent2: Dict, relation_type: str
    ) -> bool:
        """Check if a relation between two entities is reasonable."""
        if ent1["text"] == ent2["text"]:
            return False
        if len(ent1["text"]) < 2 or len(ent2["text"]) < 2:
            return False
        return True

    def _infer_relation_type(self, ent1: Dict, ent2: Dict, context: str) -> str:
        """Infer relation type based on entity types and context."""
        type1, type2 = ent1["type"], ent2["type"]
        context_lower = context.lower()

        # Basic inference based on entity types
        if type1 == "ORG" and type2 == "MONEY":
            return "manages"
        elif type1 == "PERSON" and type2 == "ORG":
            return "works_with"
        elif type1 == "ORG" and type2 == "ORG":
            if "outperform" in context_lower:
                return "outperforms"
            elif "report" in context_lower:
                return "reports_to"
            else:
                return "related_to"
        elif type1 == "ORG" and type2 == "PERCENT":
            return "has_return_of"
        elif type1 == "ORG" and type2 == "DATE":
            return "active_in"
        elif "hold" in context_lower or "holding" in context_lower:
            return "holds"
        elif "invest" in context_lower:
            return "invests_in"
        elif "outperform" in context_lower:
            return "outperforms"
        else:
            return "related_to"

    def _deduplicate_entities(self, entities: List[Dict]) -> List[Dict]:
        """Remove duplicate entities and merge their sources."""
        unique_entities = {}

        for entity in entities:
            key = (entity["text"], entity["type"])

            if key in unique_entities:
                # Merge sources
                unique_entities[key]["sources"].extend(entity["sources"])
            else:
                unique_entities[key] = entity

        return list(unique_entities.values())

    def build_knowledge_graph_with_sources(
        self, entities: List[Dict], relations: List[Dict]
    ) -> nx.DiGraph:
        """Build NetworkX DiGraph with source information."""
        G = nx.DiGraph()

        # Add nodes with entity information
        for entity in entities:
            G.add_node(entity["text"], type=entity["type"], sources=entity["sources"])

        # Add edges with relation and source information
        for relation in relations:
            if relation["subject"] in G and relation["object"] in G:
                # Check if edge already exists
                if G.has_edge(relation["subject"], relation["object"]):
                    # Merge sources for existing edge
                    current_data = G[relation["subject"]][relation["object"]]
                    if "sources" in current_data:
                        current_data["sources"].extend(relation["sources"])
                    else:
                        current_data["sources"] = relation["sources"]
                else:
                    # Create new edge with sources
                    G.add_edge(
                        relation["subject"],
                        relation["object"],
                        relation=relation["relation"],
                        sources=relation["sources"],
                    )

        return G

    def print_statistics(
        self, entities: List[Dict], relations: List[Dict], G: nx.DiGraph
    ):
        """Print clean statistics and summary."""
        print("\n" + "=" * 50)
        print("EXTRACTION SUMMARY")
        print("=" * 50)

        # Entity statistics
        entity_types = {}
        for entity in entities:
            ent_type = entity["type"]
            entity_types[ent_type] = entity_types.get(ent_type, 0) + 1

        print(f"\n📊 ENTITIES: {len(entities)} total")
        for ent_type, count in sorted(
            entity_types.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"   - {ent_type}: {count}")

        # Relation statistics
        relation_types = {}
        for relation in relations:
            rel_type = relation["relation"]
            relation_types[rel_type] = relation_types.get(rel_type, 0) + 1

        print(f"\n🔗 RELATIONS: {len(relations)} total")
        for rel_type, count in sorted(
            relation_types.items(), key=lambda x: x[1], reverse=True
        )[:10]:  # Top 10
            print(f"   - {rel_type}: {count}")

        # OCR statistics
        ocr_entities = sum(
            1
            for entity in entities
            if any(source.get("is_ocr", False) for source in entity["sources"])
        )
        ocr_relations = sum(
            1
            for relation in relations
            if any(source.get("is_ocr", False) for source in relation["sources"])
        )

        if ocr_entities > 0 or ocr_relations > 0:
            print("\n📷 OCR EXTRACTED:")
            print(f"   - Entities from OCR: {ocr_entities}")
            print(f"   - Relations from OCR: {ocr_relations}")

        # Knowledge graph statistics
        print("\n📈 KNOWLEDGE GRAPH:")
        print(f"   - Nodes: {G.number_of_nodes()}")
        print(f"   - Edges: {G.number_of_edges()}")
        if G.number_of_nodes() > 1:
            print(f"   - Density: {nx.density(G):.4f}")

        # Sample relations
        if relations:
            print("\n🔍 SAMPLE RELATIONS (first 10):")
            for i, relation in enumerate(relations[:10], 1):
                sources = relation["sources"]
                page_nums = list(set(s["page_number"] for s in sources))
                ocr_info = (
                    " [OCR]" if any(s.get("is_ocr", False) for s in sources) else ""
                )
                print(
                    f"   {i}. {relation['subject']} --{relation['relation']}--> {relation['object']}{ocr_info} (pages: {page_nums})"
                )

        # Most connected entities
        if G.number_of_edges() > 0:
            print("\n⭐ MOST CONNECTED ENTITIES:")
            degrees = {}
            for node in G.nodes():
                in_degree = len(list(G.predecessors(node)))
                out_degree = len(list(G.successors(node)))
                degrees[node] = in_degree + out_degree

            top_entities = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
            for entity, degree in top_entities:
                entity_type = G.nodes[entity].get("type", "unknown")
                print(f"   - {entity} [{entity_type}]: {degree} connections")

    def save_results(
        self,
        entities: List[Dict],
        relations: List[Dict],
        G: nx.DiGraph,
        output_dir: str = "output",
    ):
        """Save results to files."""
        Path(output_dir).mkdir(exist_ok=True)

        # Save detailed JSON
        export_data = {
            "entities": entities,
            "relations": relations,
            "graph_stats": {
                "nodes": G.number_of_nodes(),
                "edges": G.number_of_edges(),
                "ocr_enabled": self.use_ocr,
            },
        }

        with open(f"{output_dir}/extraction_report.json", "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        # Save graph for visualization
        if G.number_of_edges() > 0:
            graph_data = nx.node_link_data(G)
            with open(f"{output_dir}/knowledge_graph.json", "w", encoding="utf-8") as f:
                json.dump(graph_data, f, indent=2)

        print(f"\n💾 Results saved to {output_dir}/")

    def query_entity(self, G: nx.DiGraph, query: str):
        """Simple entity query with clean output."""
        matches = [node for node in G.nodes() if query.lower() in node.lower()]

        if matches:
            for match in matches[:2]:  # Show first 2 matches
                print(f"\n🔎 '{match}':")

                # Outgoing relations
                outgoing = list(G.out_edges(match, data=True))
                if outgoing:
                    print("   Outgoing:")
                    for _, target, data in outgoing[:3]:  # Show first 3
                        sources = data.get("sources", [])
                        page_nums = list(set(s["page_number"] for s in sources))
                        ocr_info = (
                            " [OCR]"
                            if any(s.get("is_ocr", False) for s in sources)
                            else ""
                        )
                        print(
                            f"     --{data.get('relation', 'unknown')}--> {target}{ocr_info} (pages: {page_nums})"
                        )

                # Incoming relations
                incoming = list(G.in_edges(match, data=True))
                if incoming:
                    print("   Incoming:")
                    for source, _, data in incoming[:3]:  # Show first 3
                        sources = data.get("sources", [])
                        page_nums = list(set(s["page_number"] for s in sources))
                        ocr_info = (
                            " [OCR]"
                            if any(s.get("is_ocr", False) for s in sources)
                            else ""
                        )
                        print(
                            f"     {source} --{data.get('relation', 'unknown')}-->{ocr_info} (pages: {page_nums})"
                        )
        else:
            print(f"\n❌ No matches for '{query}'")


def main():
    if len(sys.argv) < 2:
        print("Usage: python extractor.py <pdf_path>")
        print("No PDF provided, using sample data for demonstration...")
        pdf_path = "sample.pdf"
    else:
        pdf_path = sys.argv[1]

    # 用户可以选择是否启用OCR
    use_ocr = True
    if len(sys.argv) > 2:
        use_ocr = sys.argv[2].lower() in ["true", "1", "yes", "y"]

    print(f"🔧 Configuration: OCR {'ENABLED' if use_ocr else 'DISABLED'}")

    # Initialize extractor
    extractor = PDFRelationExtractor(use_ocr=use_ocr)

    # Extract elements with page information
    elements = extractor.extract_elements_from_pdf(pdf_path)
    print(f"📄 Processed {len(elements)} elements from PDF")

    # Extract entities and relations with source tracking
    entities, relations = extractor.extract_entities_relations_with_sources(elements)

    # Build knowledge graph with sources
    G = extractor.build_knowledge_graph_with_sources(entities, relations)

    # Print clean statistics
    extractor.print_statistics(entities, relations, G)

    # Save results
    extractor.save_results(entities, relations, G)

    # Demo queries
    print("\n" + "=" * 50)
    print("QUICK QUERIES")
    print("=" * 50)
    extractor.query_entity(G, "hedge")
    extractor.query_entity(G, "SEC")
    extractor.query_entity(G, "S&P500")


if __name__ == "__main__":
    main()
