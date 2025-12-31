import { NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";

export interface Comment {
  id: number;
  commenter_id: string;
  comment_id: string;
  post_id: string;
  comment_content: string;
  cleaned_content: string;
  num_emojis: number;
  num_caps_words: number;
  num_unicode_chars: number;
  contains_media: boolean;
  contains_link: boolean;
  num_tagged_people: number;
  tagged_grok: boolean;
  used_slang: boolean;
  sentiment_label: string;
  sentiment_prob: number;
  irony_label: string;
  irony_prob: number;
  hate_label: string;
  hate_prob: number;
  offensive_label: string;
  offensive_prob: number;
  // Empath categories (selected key ones)
  hate: number;
  aggression: number;
  violence: number;
  anger: number;
  rage: number;
  negative_emotion: number;
  positive_emotion: number;
  politics: number;
  government: number;
  swearing_terms: number;
}

export interface ProcessedData {
  comments: Comment[];
  stats: {
    totalComments: number;
    uniqueUsers: number;
    hateComments: number;
    offensiveComments: number;
    grokResponses: number;
    sentimentBreakdown: {
      positive: number;
      negative: number;
      neutral: number;
    };
    topUsers: { user: string; count: number }[];
  };
}

function parseBoolean(value: string): boolean {
  return value === "True" || value === "true" || value === "1";
}

function parseFloat(value: string): number {
  const num = Number.parseFloat(value);
  return Number.isNaN(num) ? 0 : num;
}

export async function GET() {
  try {
    const filePath = path.join(
      process.cwd(),
      "public",
      "final_merged_data_nlp.csv"
    );
    const fileContent = await fs.readFile(filePath, "utf-8");

    const lines = fileContent.split("\n");
    const headers = lines[0].split(",");

    // Find column indices
    const getIndex = (name: string) => headers.indexOf(name);

    const indices = {
      commenter_id: getIndex("commenter_id"),
      comment_id: getIndex("comment_id"),
      post_id: getIndex("post_id"),
      comment_content: getIndex("comment_content"),
      cleaned_content: getIndex("cleaned_content"),
      num_emojis: getIndex("num_emojis"),
      num_caps_words: getIndex("num_caps_words"),
      num_unicode_chars: getIndex("num_unicode_chars"),
      contains_media: getIndex("contains_media"),
      contains_link: getIndex("contains_link"),
      num_tagged_people: getIndex("num_tagged_people"),
      tagged_grok: getIndex("tagged_grok"),
      used_slang: getIndex("used_slang"),
      sentiment_label: getIndex("sentiment_label"),
      sentiment_prob: getIndex("sentiment_prob"),
      irony_label: getIndex("irony_label"),
      irony_prob: getIndex("irony_prob"),
      hate_label: getIndex("hate_label"),
      hate_prob: getIndex("hate_prob"),
      offensive_label: getIndex("offensive_label"),
      offensive_prob: getIndex("offensive_prob"),
      hate: getIndex("hate"),
      aggression: getIndex("aggression"),
      violence: getIndex("violence"),
      anger: getIndex("anger"),
      rage: getIndex("rage"),
      negative_emotion: getIndex("negative_emotion"),
      positive_emotion: getIndex("positive_emotion"),
      politics: getIndex("politics"),
      government: getIndex("government"),
      swearing_terms: getIndex("swearing_terms"),
    };

    const comments: Comment[] = [];
    const userCounts: Record<string, number> = {};
    let hateCount = 0;
    let offensiveCount = 0;
    let grokCount = 0;
    const sentimentCounts = { positive: 0, negative: 0, neutral: 0 };

    // Process each line (skip header, limit to reasonable amount for frontend)
    const maxRows = 50000; // Limit for performance
    for (let i = 1; i < Math.min(lines.length, maxRows + 1); i++) {
      const line = lines[i];
      if (!line.trim()) continue;

      // Smart CSV parsing (handle commas in quoted strings)
      const values: string[] = [];
      let current = "";
      let inQuotes = false;

      for (const char of line) {
        if (char === '"') {
          inQuotes = !inQuotes;
        } else if (char === "," && !inQuotes) {
          values.push(current);
          current = "";
        } else {
          current += char;
        }
      }
      values.push(current);

      const commenter_id = values[indices.commenter_id] || "";
      const sentiment_label = values[indices.sentiment_label] || "neutral";
      const hate_label = values[indices.hate_label] || "NOT-HATE";
      const offensive_label =
        values[indices.offensive_label] || "non-offensive";
      const tagged_grok = parseBoolean(values[indices.tagged_grok] || "");

      // Skip empty/invalid rows
      if (!commenter_id || commenter_id.length > 50) continue;

      // Track stats
      userCounts[commenter_id] = (userCounts[commenter_id] || 0) + 1;

      if (hate_label === "HATE") hateCount++;
      if (offensive_label === "offensive") offensiveCount++;
      if (tagged_grok || commenter_id === "grok") grokCount++;

      if (sentiment_label === "positive") sentimentCounts.positive++;
      else if (sentiment_label === "negative") sentimentCounts.negative++;
      else sentimentCounts.neutral++;

      comments.push({
        id: i,
        commenter_id,
        comment_id: values[indices.comment_id] || "",
        post_id: values[indices.post_id] || "",
        comment_content: values[indices.comment_content] || "",
        cleaned_content: values[indices.cleaned_content] || "",
        num_emojis: parseFloat(values[indices.num_emojis]),
        num_caps_words: parseFloat(values[indices.num_caps_words]),
        num_unicode_chars: parseFloat(values[indices.num_unicode_chars]),
        contains_media: parseBoolean(values[indices.contains_media] || ""),
        contains_link: parseBoolean(values[indices.contains_link] || ""),
        num_tagged_people: parseFloat(values[indices.num_tagged_people]),
        tagged_grok,
        used_slang: parseBoolean(values[indices.used_slang] || ""),
        sentiment_label,
        sentiment_prob: parseFloat(values[indices.sentiment_prob]),
        irony_label: values[indices.irony_label] || "non_irony",
        irony_prob: parseFloat(values[indices.irony_prob]),
        hate_label,
        hate_prob: parseFloat(values[indices.hate_prob]),
        offensive_label,
        offensive_prob: parseFloat(values[indices.offensive_prob]),
        hate: parseFloat(values[indices.hate]),
        aggression: parseFloat(values[indices.aggression]),
        violence: parseFloat(values[indices.violence]),
        anger: parseFloat(values[indices.anger]),
        rage: parseFloat(values[indices.rage]),
        negative_emotion: parseFloat(values[indices.negative_emotion]),
        positive_emotion: parseFloat(values[indices.positive_emotion]),
        politics: parseFloat(values[indices.politics]),
        government: parseFloat(values[indices.government]),
        swearing_terms: parseFloat(values[indices.swearing_terms]),
      });
    }

    // Get top users
    const topUsers = Object.entries(userCounts)
      .filter(([user]) => user.length > 0 && user.length < 30)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 20)
      .map(([user, count]) => ({ user, count }));

    const data: ProcessedData = {
      comments,
      stats: {
        totalComments: comments.length,
        uniqueUsers: Object.keys(userCounts).length,
        hateComments: hateCount,
        offensiveComments: offensiveCount,
        grokResponses: grokCount,
        sentimentBreakdown: sentimentCounts,
        topUsers,
      },
    };

    return NextResponse.json(data);
  } catch (error) {
    console.error("Error reading CSV:", error);
    return NextResponse.json({ error: "Failed to load data" }, { status: 500 });
  }
}
